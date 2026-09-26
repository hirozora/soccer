#!/usr/bin/env python
"""Read-only Main-L2 attention audit of locked ECA checkpoints, validation only."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

VERSION = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(VERSION / 'src'), str(VERSION.parent / 'benchmark_unified_v1/src')]
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '1'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import numpy as np
import pandas as pd
import torch
from torch_geometric.utils import softmax

torch.set_num_threads(1)
torch.backends.mkldnn.enabled = False
torch.use_deterministic_algorithms(True)

from football_hgt_targets_v4.eca_training import ROOT, SEEDS, loader_for, load_backbone, write_json
from football_hgt_targets_v4.coverage_history_training import load_backbone as load_rotate
from football_hgt_targets_v4.spatiotemporal_edge import edge_bundle
from football_hgt_targets_v4.position_head_refit import sha256
from football_hgt_targets_v4.training import _move_batch_to_device

OUT = ROOT / 'diagnostics/anchor_age_v1'
MODES = ('base', 'constant', 'transition')
AGES = ('ends_at_anchor', 'age_1_5', 'age_6_20', 'age_gt20')
ANCHORS = ('other', 'Pass+CONTROL')


def age_bucket(age):
    if bool((age < 0).any()):
        raise ValueError('Future pair endpoint')
    return torch.where(age == 0, 0, torch.where(age <= 5, 1, torch.where(age <= 20, 2, 3)))


class AttentionCapture:
    """Observe messages without modifying their output or model parameters."""
    def __init__(self, layer):
        self.layer = layer
        self.handle = layer.register_message_forward_hook(self.capture)
        self.calls = 0

    def capture(self, module, inputs, output):
        values = inputs[0]
        score = (values['q_i'] * values['k_j']).sum(-1) * values['edge_attr']
        score = score / math.sqrt(values['q_i'].shape[-1])
        bias = getattr(module, '_edge_bias', None)
        if bias is not None:
            score = score + bias
        attention = softmax(score, values['index'], values['ptr'], values['size_i'])
        reproduced = (values['v_j'] * attention.unsqueeze(-1)).view_as(output)
        if not torch.equal(reproduced, output):
            raise RuntimeError('Observed attention does not reproduce actual messages')
        self.attention = attention.detach().cpu()
        self.destination = values['index'].detach().cpu()
        total = torch.zeros((int(values['index'].max()) + 1, attention.shape[1]))
        total.index_add_(0, self.destination, self.attention)
        active = torch.unique(self.destination)
        if active.numel() and float((total[active] - 1).abs().max()) > 2e-6:
            raise RuntimeError('Attention is not normalized over all incoming edges')
        self.calls += 1

    def close(self):
        self.handle.remove()


def pair_mass(bundle, edge_types, attention):
    maps = torch.cat([bundle['mappings'][r].cpu() for r in edge_types])
    if len(maps) != len(attention):
        raise RuntimeError('Edge ordering mismatch')
    active = maps >= 0
    result = torch.zeros((bundle['pairs'].shape[1], 4))
    result.index_add_(0, maps[active], attention[active])
    if bool((result < 0).any()) or bool((result > 1 + 2e-6).any()):
        raise RuntimeError('Invalid adjacent-pair attention mass')
    return result.numpy()


def run_seed(seed, device, out=OUT, limit=None):
    target = out / f'seed{seed}'
    target.mkdir(parents=True, exist_ok=True)
    done = target / 'completed.json'
    if done.exists():
        previous = json.loads(done.read_text())
        if previous['limit'] != limit or previous['script_sha256'] != sha256(Path(__file__)):
            raise RuntimeError('Audit provenance changed; use a separate output directory')
        return previous
    models, captures, paths, initial_hashes = {}, {}, {}, {}
    from football_hgt_targets_v4.position_head_refit import tensor_hash
    for mode in MODES:
        model, path = (load_rotate('rotate', seed, device) if mode == 'base'
                       else load_backbone(mode, seed, device))
        models[mode] = model.eval().requires_grad_(False)
        captures[mode] = AttentionCapture(model.convolutions[1])
        paths[str(path)] = sha256(path)
        initial_hashes[mode] = tensor_hash(model.state_dict())
    loader = loader_for('validation', seed, device, ROOT, workers=2 if limit is None else 0, limit=limit)
    # Each key has pair count, sample count, per-head pair sums, sample-mean sums,
    # bias sums, bias-squared sums. Long windows cannot dominate sample means.
    stats = {}
    seen = set()
    digest = hashlib.sha256()
    coverage = np.zeros((2, 5), dtype=np.int64)
    started = time.monotonic()
    try:
        for number, raw in enumerate(loader, 1):
            for sample in raw['sample_ids']:
                if sample in seen:
                    raise RuntimeError('Duplicate validation sample')
                seen.add(sample); digest.update((str(sample) + '\n').encode())
            batch = _move_batch_to_device(raw, torch.device(device))
            graph = batch['graphs']['f80']
            bundle = edge_bundle(graph)
            src, dst = bundle['pairs']
            ev = graph['event']
            anchor = ev.ptr[1:] - 1
            sample = ev.batch[dst]
            age = anchor[sample] - dst
            if not torch.equal(age, ev.source_index[anchor[sample]] - ev.source_index[dst]):
                raise RuntimeError('F80 local rank differs from source Event distance')
            bucket = age_bucket(age).cpu().numpy()
            sample = sample.cpu().numpy()
            valid = bundle['eligible'].cpu().numpy()
            if len(torch.unique(dst[bundle['eligible']])) != int(bundle['eligible'].sum()):
                raise RuntimeError('More than one unique adjacent pair per destination')
            focal = ((ev.event_type_index[anchor] == 7) & (ev.control_state_after_index[anchor] == 2)).cpu().numpy().astype(int)
            matches = batch['match_ids'].cpu().numpy()
            for label in range(2):
                coverage[label, 0] += (focal == label).sum()
                for age_id in range(4):
                    present = np.unique(sample[valid & (bucket == age_id)])
                    coverage[label, age_id + 1] += (focal[present] == label).sum()
            with torch.no_grad():
                for mode, model in models.items():
                    output = model(batch)
                    capture = captures[mode]
                    if number == 1:
                        capture.close()
                        unobserved = model(batch)
                        if not all(torch.equal(output[k], unobserved[k]) for k in output):
                            raise RuntimeError('Observation hook altered outputs')
                        capture.handle = model.convolutions[1].register_message_forward_hook(capture.capture)
                    mass = pair_mass(bundle, graph.edge_index_dict, capture.attention)
                    bias = np.zeros_like(mass) if mode == 'base' else model.convolutions[1].last_pair_bias.cpu().numpy()
                    for age_id in range(4):
                        mask = valid & (bucket == age_id)
                        counts = np.bincount(sample[mask], minlength=len(matches))
                        sums = np.stack([np.bincount(sample[mask], weights=mass[mask, head], minlength=len(matches)) for head in range(4)], -1)
                        biases = np.stack([np.bincount(sample[mask], weights=bias[mask, head], minlength=len(matches)) for head in range(4)], -1)
                        squared = np.stack([np.bincount(sample[mask], weights=bias[mask, head]**2, minlength=len(matches)) for head in range(4)], -1)
                        for row in np.flatnonzero(counts):
                            key = (mode, int(matches[row]), int(focal[row]), age_id)
                            record = stats.setdefault(key, np.zeros(18))
                            record[:2] += [counts[row], 1]
                            record[2:6] += sums[row]
                            record[6:10] += sums[row] / counts[row]
                            record[10:14] += biases[row]
                            record[14:18] += squared[row]
            if number % 10 == 0:
                write_json(target / 'progress.json', {'seed': seed, 'batches': number,
                    'samples': len(seen), 'seconds': time.monotonic() - started, 'total': len(loader.dataset)})
                print(json.dumps({'seed': seed, 'samples': len(seen), 'total': len(loader.dataset)}), flush=True)
    finally:
        for capture in captures.values():
            capture.close()
    if len(seen) != (limit or 96891):
        raise RuntimeError('Incomplete audit')
    if any(tensor_hash(m.state_dict()) != initial_hashes[name] for name, m in models.items()):
        raise RuntimeError('Audit changed model parameters')
    keys = sorted(stats)
    np.savez_compressed(target / 'match_statistics.npz',
        keys=np.array([[MODES.index(m), match, group, age] for m, match, group, age in keys]),
        values=np.stack([stats[k] for k in keys]))
    result = {'seed': seed, 'samples': len(seen), 'sample_ids_sha256': digest.hexdigest(),
        'seconds': time.monotonic() - started, 'limit': limit, 'checkpoint_hashes': paths,
        'script_sha256': sha256(Path(__file__)), 'coverage': coverage.tolist(), 'test_accessed': False,
        'parameters_unchanged': True, 'hook_outputs_exactly_equal': True}
    write_json(done, result)
    return result


def summarize(out=OUT):
    meta = [json.loads((out / f'seed{s}/completed.json').read_text()) for s in SEEDS]
    if len({m['sample_ids_sha256'] for m in meta}) != 1:
        raise RuntimeError('Seeds did not see identical ordered samples')
    data = [np.load(out / f'seed{s}/match_statistics.npz') for s in SEEDS]
    matches = sorted(set(data[0]['keys'][:, 1].tolist()))
    lookup = {m: i for i, m in enumerate(matches)}
    array = np.zeros((3, 3, len(matches), 2, 4, 18), dtype=np.float64)
    for seed_index, item in enumerate(data):
        for (mode, match, group, age), values in zip(item['keys'], item['values']):
            array[seed_index, mode, lookup[match], group, age] = values
    if not np.array_equal(array[:, :1, :, :, :, :2].repeat(3, axis=1), array[:, :, :, :, :, :2]):
        raise RuntimeError('Models have different attention populations')
    rows = []
    aggregate = array.sum(2)
    for si, seed in enumerate(SEEDS):
        for mi, mode in enumerate(MODES):
            for group in range(2):
                for age in range(4):
                    rec = aggregate[si, mi, group, age]
                    for head in range(5):
                        h = slice(None) if head == 4 else head
                        mass = (rec[6:10] / max(1, rec[1]))[h].mean()
                        rows.append({'seed': seed, 'mode': mode, 'anchor': ANCHORS[group], 'age': AGES[age],
                            'head': 'mean' if head == 4 else str(head), 'pairs': int(rec[0]), 'samples': int(rec[1]),
                            'sample_mean_mass': mass, 'pair_mean_mass': (rec[2:6] / max(1, rec[0]))[h].mean(),
                            'bias_mean': (rec[10:14] / max(1, rec[0]))[h].mean()})
    frame = pd.DataFrame(rows)
    frame.to_csv(out / 'attention_per_seed.csv', index=False)
    frame.groupby(['mode', 'anchor', 'age', 'head']).agg(
        mean_mass=('sample_mean_mass', 'mean'), std_seed=('sample_mean_mass', 'std'),
        pair_mean_mass=('pair_mean_mass', 'mean'), bias_mean=('bias_mean', 'mean'),
        samples=('samples', 'first'), pairs=('pairs', 'first')).to_csv(out / 'attention_summary.csv')
    draws = np.random.default_rng(20260715).multinomial(len(matches), np.full(len(matches), 1/len(matches)), 10000)
    pairs = [('constant', 'base'), ('transition', 'base'), ('transition', 'constant')]
    differences, interactions = [], []
    for candidate, reference in pairs:
        ci, ri = MODES.index(candidate), MODES.index(reference)
        observed_groups, bootstrap_groups = {}, {}
        for group in range(2):
            for age in range(4):
                a = array[:, ci, :, group, age]
                b = array[:, ri, :, group, age]
                weights = a[:, :, 1]
                delta_sum = (a[:, :, 6:10] - b[:, :, 6:10]).mean(-1)
                observed = delta_sum.sum(1) / np.maximum(weights.sum(1), 1)
                den = np.einsum('bm,sm->bs', draws, weights)
                sampled = np.einsum('bm,sm->bs', draws, delta_sum) / np.maximum(den, 1)
                observed_groups[group, age] = observed
                bootstrap_groups[group, age] = sampled.mean(1)
                low, high = np.quantile(sampled.mean(1), [.025, .975])
                differences.append({'comparison': f'{candidate}-{reference}', 'anchor': ANCHORS[group],
                    'age': AGES[age], 'mean_difference': observed.mean(), 'ci_low': low, 'ci_high': high,
                    'seed_differences': observed.tolist()})
        contrasts = {
            'focal_anchor_minus_focal_old': [(1, 0, 1), (1, 3, -1)],
            'focal_anchor_minus_other_anchor': [(1, 0, 1), (0, 0, -1)],
            'focal_vs_other_anchor_old_interaction': [(1, 0, 1), (1, 3, -1), (0, 0, -1), (0, 3, 1)]}
        for name, terms in contrasts.items():
            observed = sum(sign * observed_groups[g, a] for g, a, sign in terms)
            sampled = sum(sign * bootstrap_groups[g, a] for g, a, sign in terms)
            low, high = np.quantile(sampled, [.025, .975])
            interactions.append({'comparison': f'{candidate}-{reference}', 'contrast': name,
                'mean_difference': observed.mean(), 'seed_differences': observed.tolist(),
                'ci_low': low, 'ci_high': high})
    write_json(out / 'paired_differences.json', differences)
    write_json(out / 'anchor_age_interactions.json', interactions)
    summary = frame[frame['head'] == 'mean'].groupby(['anchor', 'age', 'mode']).sample_mean_mass.mean().unstack('mode')
    lines = ['# ECA Anchor/Age Attention Audit', '',
        f"Validation samples per seed: {meta[0]['samples']}; three frozen checkpoints per mode.",
        'No training or test access. Pair age is destination Event local rank relative to anchor.',
        'Mass sums next+gap attention after normalization over ALL incoming edges, averaged across four heads.',
        'Primary averages give each sample equal weight within its age bucket. Pair-weighted means are supplementary.',
        '', '| Anchor | Pair age | Base | Constant | Transition |', '|---|---|---:|---:|---:|']
    for (anchor, age), row in summary.iterrows():
        lines.append(f"| {anchor} | {age} | {row['base']:.4f} | {row['constant']:.4f} | {row['transition']:.4f} |")
    lines += ['', 'CIs are descriptive paired match-cluster intervals (10000 shared draws).',
        'Differences across separately trained models are not isolated causal effects of controller bias.',
        'No change to the validation selection, test gate, or official Rotate baseline.',
        'No-context/period-crossing pairs are excluded from the controlled-edge population; coverage is recorded per seed.',
        'Attention does not measure information redundancy, task-gradient dominance, or feature sufficiency by itself.']
    (out / 'README.md').write_text('\n'.join(lines) + '\n')
    write_json(out / 'completed.json', {'status': 'completed', 'seeds': SEEDS,
        'samples_per_seed': meta[0]['samples'], 'validation_only': True, 'training_performed': False,
        'script_sha256': sha256(Path(__file__))})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, choices=SEEDS)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--out', type=Path, default=OUT)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--summarize', action='store_true')
    args = parser.parse_args()
    out = args.out.resolve(); out.mkdir(parents=True, exist_ok=True)
    if args.seed:
        run_seed(args.seed, args.device, out, args.limit)
        return
    if args.summarize:
        summarize(out)
        return
    running = []
    try:
        for gpu, seed in enumerate(SEEDS):
            log = (out / f'seed{seed}.log').open('a')
            command = [sys.executable, str(Path(__file__).resolve()), '--seed', str(seed),
                       '--device', f'cuda:{gpu}', '--out', str(out)]
            if args.limit:
                command += ['--limit', str(args.limit)]
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            running.append((process, log))
        write_json(out / 'status.json', {'status': 'running', 'pid': os.getpid(),
            'workers': [p.pid for p, _ in running], 'started_at': time.time()})
        codes = [p.wait() for p, _ in running]
        if any(codes):
            raise RuntimeError(f'Audit workers failed: {codes}')
        summarize(out)
        write_json(out / 'status.json', {'status': 'completed', 'finished_at': time.time()})
    except BaseException as error:
        write_json(out / 'status.json', {'status': 'failed', 'error': repr(error)})
        raise
    finally:
        for process, log in running:
            if process.poll() is None:
                process.terminate(); process.wait()
            log.close()


if __name__ == '__main__':
    main()
