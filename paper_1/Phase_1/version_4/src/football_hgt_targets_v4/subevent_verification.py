"""Preflight equivalence, exact Rotate sampling and CUDA measurements."""
import json
import hashlib
import time

import numpy as np
import torch

from football_benchmark.protocol import ProtocolArtifacts
from .constants import FEASIBILITY_ARTIFACT, POSSESSION_GRAPH_ROOT
from football_benchmark.data import load_records
from .coverage_history_training import load_backbone as load_rotate
from .subevent_auxiliary import SubeventAuxiliaryHGT, public_state, MODES, label, SUBTYPES, inference_model
from .subevent_training import (ROOT, ROTATE_ROOT, SEEDS, loader_for, rotation_loader, sources,
                          train_subevent, load_backbone, run_dir, write_json)
from .fixed_budget_loss import fixed_budget_loss
from .fixed_budget_study import ALL_TASKS
from .model import build_partial_l2_model
from .position_head_refit import tensor_hash
from .training import set_seed, _move_batch_to_device


def audit(root=ROOT):
    original = json.loads((ROTATE_ROOT / 'verification/data_audit.json').read_text())
    assert original['train_targets'] == 449025 and original['epochs'][15]['unique_targets_so_far'] == 449025
    loader, _ = rotation_loader('fine', SEEDS[0], 'cpu', root, workers=0)
    hashes = []
    for seed in SEEDS:
        old = json.loads((ROTATE_ROOT / 'coverage/rotate' / f'seed{seed}' / 'history.json').read_text())
        assert len(old) == 24 and old[-1]['steps'] == 3192
        for e, row in enumerate(old, 1):
            actual = loader.dataset.plan_hash(e)
            assert actual == row['target_plan_sha256']
            hashes.append({'seed': seed, 'epoch': e, 'sha256': actual})
    label_audit(root, loader.dataset)
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
        model = SubeventAuxiliaryHGT(artifacts, mode).to(device).eval()
        assert tensor_hash(public_state(model)) == digest
        with torch.no_grad():
            actual = model(batch)
            error = max(float((actual[k] - expected[k]).abs().max()) for k in expected)
            difference = abs(float(fixed_budget_loss(actual, batch, artifacts, ALL_TASKS)[0] - loss))
        assert error < 1e-6 and difference < 1e-6
        errors[mode] = {'output': error, 'loss': difference}
        del model
    del base, actual, expected, batch
    torch.cuda.empty_cache()
    costs = {}
    for mode in ('base', *MODES):
        set_seed(seed)
        model = (build_partial_l2_model(artifacts) if mode == 'base' else SubeventAuxiliaryHGT(artifacts, mode)).to(device).eval()
        costs[mode] = benchmark(model, raw, device)
        del model
        torch.cuda.empty_cache()
    out = root / 'smoke' / hashlib.sha256(json.dumps(source, sort_keys=True).encode()).hexdigest()[:12]
    results = [train_subevent(mode, seed, device, out, workers=0, smoke=True) for mode in MODES]
    result = {'passed': True, 'sources': sources(), 'device': device, 'equivalence': errors,
              'runs': results, 'preliminary_efficiency': costs}
    write_json(root / 'verification/smoke.json', result)
    return result


def label_audit(root, dataset):
    artifacts = ProtocolArtifacts.load(FEASIBILITY_ARTIFACT)
    result = {'subtypes': SUBTYPES, 'splits': {}, 'test_accessed': False}
    expected = {
        'train': [26626,37841,37717,21377,3341,8168,23617,8607,1734,14743,17473,7134,175517,4206],
        'validation': [5750,8126,8127,4797,762,1787,5258,1859,343,3467,3812,1604,37139,831]}
    rotated = np.zeros(14, dtype=np.int64)
    for split in ('train', 'validation'):
        counts = np.zeros(14, dtype=np.int64)
        total = 0
        for number, record in enumerate(load_records(split, graph_root=POSSESSION_GRAPH_ROOT)):
            graph = torch.load(record.graph_path, map_location='cpu', weights_only=False)
            event = graph['node_stores']['event']
            # Supervise immediate successors, including the first event of the next period.
            indices = range(1, record.num_events)
            types = event['event_type_index'].tolist()
            subtypes = event['subevent_type_index'].tolist()
            ids = np.array([label(artifacts.event_type_ids[types[i]],
                artifacts.subevent_type_ids[subtypes[i]])[0] for i in indices])
            counts += np.bincount(ids[ids >= 0], minlength=14)
            total += len(ids)
            if split == 'train':
                if dataset.records[number].match_id != record.match_id:
                    raise RuntimeError('Rotation audit record ordering mismatch')
                repeated = ids[dataset.tables[number].numpy().reshape(-1)]
                rotated += np.bincount(repeated[repeated >= 0], minlength=14)
        if counts.tolist() != expected[split] or total != {'train':449025, 'validation':96891}[split]:
            raise RuntimeError(f'Unexpected raw subtype coverage: {split}, {counts.tolist()}, {total}')
        result['splits'][split] = {'targets': total, 'eligible': int(counts.sum()), 'counts': counts.tolist()}
    result['rotate_24_supervision_counts'] = rotated.tolist()
    write_json(root / 'verification/subtype_counts.json', result)


@torch.no_grad()
def diagnostics(mode, seed, device, root=ROOT):
    if mode != 'base' and not json.loads((run_dir(mode, seed, root) / 'result.json').read_text())['eligible']:
        return
    model, _ = load_rotate('rotate', seed, device) if mode == 'base' else load_backbone(mode, seed, device, root)
    loader = loader_for('validation', seed, device, root, workers=0, limit=256)
    samples = [loader.dataset[i] for i in range(len(loader.dataset))]
    started = time.perf_counter()
    raw = loader.collate_fn(samples)
    cpu = time.perf_counter() - started
    if mode != 'base':
        original = model(_move_batch_to_device(raw, torch.device(device)))
        stripped = inference_model(ProtocolArtifacts.load(FEASIBILITY_ARTIFACT), model.state_dict()).to(device).eval()
        restored = stripped(_move_batch_to_device(raw, torch.device(device)))
        error = max(float((restored[k] - original[k]).abs().max()) for k in restored)
        assert error < 1e-6
        del model
        model = stripped
    cost = benchmark(model, raw, device)
    cost.update(cpu_collate_seconds=cpu, auxiliary_head_removed=mode != 'base',
                inference_equivalence_error=0. if mode == 'base' else error)
    cost['full_inference_seconds'] = cpu + cost['transfer_forward_seconds']
    write_json(root / 'diagnostics' / mode / f'seed{seed}' / 'efficiency.json', cost)

