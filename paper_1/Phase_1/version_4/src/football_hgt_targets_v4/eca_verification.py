"""Preflight equivalence, exact Rotate sampling and CUDA measurements."""
import json
import hashlib
import time

import numpy as np
import torch

from football_benchmark.protocol import ProtocolArtifacts
from .constants import FEASIBILITY_ARTIFACT
from .coverage_history_training import load_backbone as load_rotate
from .eca_transition import TransitionECAHGT, public_state, MODES
from .eca_training import (ROOT, ROTATE_ROOT, SEEDS, loader_for, rotation_loader, sources,
                          train_eca, load_backbone, run_dir, write_json)
from .fixed_budget_loss import fixed_budget_loss
from .fixed_budget_study import ALL_TASKS
from .model import build_partial_l2_model
from .position_head_refit import tensor_hash
from .training import set_seed, _move_batch_to_device


def audit(root=ROOT):
    original = json.loads((ROTATE_ROOT / 'verification/data_audit.json').read_text())
    assert original['train_targets'] == 449025 and original['epochs'][15]['unique_targets_so_far'] == 449025
    loader, _ = rotation_loader('transition', SEEDS[0], 'cpu', root, workers=0)
    hashes = []
    for seed in SEEDS:
        old = json.loads((ROTATE_ROOT / 'coverage/rotate' / f'seed{seed}' / 'history.json').read_text())
        assert len(old) == 24 and old[-1]['steps'] == 3192
        for e, row in enumerate(old, 1):
            actual = loader.dataset.plan_hash(e)
            assert actual == row['target_plan_sha256']
            hashes.append({'seed': seed, 'epoch': e, 'sha256': actual})
    write_json(root / 'verification/target_plan.json', {'passed': True, 'hashes': hashes,
               'source_coverage_audit': original, 'sources': sources()})


def sync(device):
    if str(device).startswith('cuda'):
        torch.cuda.synchronize(device)


@torch.no_grad()
def benchmark(model, raw, device, repeats=12):
    batch = _move_batch_to_device(raw, torch.device(device))
    model.eval()
    for _ in range(3):
        model(batch)
    sync(device)
    if str(device).startswith('cuda'):
        torch.cuda.reset_peak_memory_stats(device)
    forward, h2d, full = [], [], []
    for _ in range(repeats):
        start = time.perf_counter()
        moved = _move_batch_to_device(raw, torch.device(device))
        sync(device)
        mid = time.perf_counter()
        model(moved)
        sync(device)
        end = time.perf_counter()
        h2d.append(mid - start); forward.append(end - mid); full.append(end - start)
    return {'forward_seconds': float(np.median(forward)), 'h2d_seconds': float(np.median(h2d)),
            'transfer_forward_seconds': float(np.median(full)),
            'samples_per_second_model': len(raw['sample_ids']) / float(np.median(forward)),
            'peak_memory_bytes': torch.cuda.max_memory_allocated(device) if str(device).startswith('cuda') else None,
            'parameters': sum(p.numel() for p in model.parameters()), 'samples': len(raw['sample_ids']),
            'hgt_calls_per_batch': 3}


def smoke(root=ROOT, device='cuda:0'):
    if not str(device).startswith('cuda') or not torch.cuda.is_available():
        raise RuntimeError('CUDA smoke required')
    source = sources()
    previous = root / 'verification/smoke.json'
    if previous.exists():
        record = json.loads(previous.read_text())
        if record['sources'] == source and record['passed']:
            return record
    artifacts = ProtocolArtifacts.load(FEASIBILITY_ARTIFACT)
    seed = SEEDS[0]
    raw = next(iter(loader_for('validation', seed, device, root, workers=0, limit=256)))
    batch = _move_batch_to_device(raw, torch.device(device))
    set_seed(seed)
    base = build_partial_l2_model(artifacts).to(device).eval()
    digest = tensor_hash(base.state_dict())
    with torch.no_grad():
        expected = base(batch)
        loss = fixed_budget_loss(expected, batch, artifacts, ALL_TASKS)[0]
    errors = {}
    for mode in MODES:
        set_seed(seed)
        model = TransitionECAHGT(artifacts, mode).to(device).eval()
        assert tensor_hash(public_state(model)) == digest
        with torch.no_grad():
            actual = model(batch)
            error = max(float((actual[k] - expected[k]).abs().max()) for k in actual)
            difference = abs(float(fixed_budget_loss(actual, batch, artifacts, ALL_TASKS)[0] - loss))
        assert error < 1e-6 and difference < 1e-6
        errors[mode] = {'output': error, 'loss': difference}
        del model
    del base, actual, expected, batch
    torch.cuda.empty_cache()
    costs = {}
    for mode in ('base', *MODES):
        set_seed(seed)
        model = (build_partial_l2_model(artifacts) if mode == 'base' else TransitionECAHGT(artifacts, mode)).to(device).eval()
        costs[mode] = benchmark(model, raw, device)
        del model
        torch.cuda.empty_cache()
    out = root / 'smoke' / hashlib.sha256(json.dumps(source, sort_keys=True).encode()).hexdigest()[:12]
    results = [train_eca(mode, seed, device, out, workers=0, smoke=True) for mode in MODES]
    result = {'passed': True, 'sources': sources(), 'device': device, 'equivalence': errors,
              'runs': results, 'preliminary_efficiency': costs}
    write_json(root / 'verification/smoke.json', result)
    return result


@torch.no_grad()
def diagnostics(mode, seed, device, root=ROOT):
    if mode != 'base' and not json.loads((run_dir(mode, seed, root) / 'result.json').read_text())['eligible']:
        return
    model, _ = (load_rotate('rotate', seed, device) if mode == 'base' else load_backbone(mode, seed, device, root))
    loader = loader_for('validation', seed, device, root, workers=0, limit=256)
    samples = [loader.dataset[i] for i in range(len(loader.dataset))]
    elapsed = []
    for _ in range(5):
        start = time.perf_counter()
        raw = loader.collate_fn(samples)
        elapsed.append(time.perf_counter() - start)
    cost = benchmark(model, raw, device)
    cost['cpu_collate_seconds'] = float(np.median(elapsed))
    cost['full_inference_seconds'] = cost['cpu_collate_seconds'] + cost['transfer_forward_seconds']
    cost['samples_per_second_full'] = len(samples) / cost['full_inference_seconds']
    cost['scope'] = 'Same fixed 256 validation samples, raw backbone; shared postprocessing excluded'
    base_path = root / 'diagnostics/base' / f'seed{seed}' / 'efficiency.json'
    if mode != 'base' and base_path.exists():
        base = json.loads(base_path.read_text())
        cost['relative_forward'] = cost['forward_seconds'] / base['forward_seconds']
        cost['relative_memory'] = cost['peak_memory_bytes'] / base['peak_memory_bytes']
        cost['low_cost'] = cost['relative_forward'] <= 1.20 and cost['relative_memory'] <= 1.10
    out = root / 'diagnostics' / mode / f'seed{seed}'
    write_json(out / 'efficiency.json', cost)
    if mode == 'base':
        del model
        torch.cuda.empty_cache()
        deployed_efficiency(mode, seed, raw, cost['cpu_collate_seconds'], device, root)
        return
    layer = model.convolutions[1]
    layer.capture_attention = True
    rows = []
    for raw in loader_for('validation', seed, device, root, workers=0, limit=1024):
        model(_move_batch_to_device(raw, torch.device(device)))
        bundle = model.last_bundle
        features = bundle['features']
        valid = bundle['eligible']
        masks = {'all': valid, 'same_possession': valid & (features[:, 7] > 0),
                 'different_or_unknown_possession': valid & (features[:, 7] == 0)}
        seconds = features[:, 0].mul(np.log(7201)).expm1()
        meters = features[:, 3] * np.hypot(105, 68)
        for label, values, bounds in [('seconds', seconds, [0, 2, 5, 15, float('inf')]),
                                      ('meters', meters, [0, 10, 30, 60, float('inf')])]:
            for lo, hi in zip(bounds, bounds[1:]):
                masks[f'{label}_{lo}_{hi}'] = valid & (values >= lo) & (values < hi)
        for label, start, width in [('source_control', 10, 4), ('target_control', 14, 4),
                                     ('source_role', 18, 6), ('target_role', 24, 6)]:
            for category in range(width):
                masks[f'{label}_{category}'] = valid & (features[:, start + category] > 0)
        # Sum next+gap mass per destination after normalization over ALL incoming edges.
        maps = torch.cat([bundle['mappings'][r] for r in raw['graphs']['f80'].edge_index_dict]).to(device)
        eligible = maps >= 0
        pair_mass = layer.last_attention.new_zeros((len(features), 4))
        # No gradients here; CPU accumulation avoids nondeterministic CUDA atomics.
        mass = pair_mass.cpu()
        mass.index_add_(0, maps[eligible].cpu(), layer.last_attention[eligible].cpu())
        for label, mask in masks.items():
            mask = mask.cpu()
            n = int(mask.sum())
            if not n:
                continue
            bias = layer.last_pair_bias.cpu()[mask]
            rows.append({'group': label, 'pairs': n,
                         'bias_sum': bias.sum(0).tolist(), 'bias_squared_sum': bias.square().sum(0).tolist(),
                         'attention_mass_sum': mass[mask].sum(0).tolist()})
    write_json(out / 'attention.json', {'rows': rows, 'scope': 'fixed first 1024 validation samples',
               'attention_denominator': 'all incoming HGT edges', 'heads_have_no_assigned_semantics': True})
    layer.capture_attention = False
    del model, layer, bundle, pair_mass, features, masks, mass
    torch.cuda.empty_cache()
    # Use the same original 256 samples, not the last diagnostics batch.
    raw = loader.collate_fn(samples)
    deployed_efficiency(mode, seed, raw, cost['cpu_collate_seconds'], device, root)


@torch.no_grad()
def deployed_efficiency(mode, seed, raw, cpu_seconds, device, root):
    from .coverage_history_training import load_deployed as rotate_deployed
    from .eca_training import load_deployed
    model = (rotate_deployed('rotate', seed, device) if mode == 'base'
             else load_deployed(mode, seed, device, root))
    cost = benchmark(model, raw, device)
    cost['cpu_collate_seconds'] = cpu_seconds
    cost['full_inference_seconds'] = cpu_seconds + cost['transfer_forward_seconds']
    cost['samples_per_second_full'] = len(raw['sample_ids']) / cost['full_inference_seconds']
    cost['scope'] = 'Same 256 validation samples, including Position Refit and TC lambda=1.5'
    base_path = root / 'diagnostics/base' / f'seed{seed}' / 'deployed_efficiency.json'
    if mode != 'base' and base_path.exists():
        base = json.loads(base_path.read_text())
        cost['relative_forward'] = cost['forward_seconds'] / base['forward_seconds']
        cost['relative_memory'] = cost['peak_memory_bytes'] / base['peak_memory_bytes']
        cost['relative_full_inference'] = cost['full_inference_seconds'] / base['full_inference_seconds']
    write_json(root / 'diagnostics' / mode / f'seed{seed}' / 'deployed_efficiency.json', cost)
