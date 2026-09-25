"""Independent fixed-budget training and checkpoint-specific deployment outputs."""
from dataclasses import asdict
from pathlib import Path
import json
import time
import numpy as np
import pandas as pd
import torch
from football_benchmark.protocol import ProtocolArtifacts
from .constants import EXPERIMENT_ROOT, FEASIBILITY_ARTIFACT, SAMPLE_PLAN, POSSESSION_GRAPH_ROOT
from .spatiotemporal_edge import SpatiotemporalHGT
from .fixed_budget_training import (FixedBudgetConfig, _loader_config, _common_hash,
    _train_epoch, evaluate_fixed_budget, _rng_state, _restore_rng,
    _gradient_diagnostics, _partial_layer_gradient_diagnostics, _guard_metrics, guarded_core_eligible)
from .five_task_training import _loader, _merge_prediction_frames
from .fixed_budget_study import ALL_TASKS, training_dir as five_dir
from .partial_sharing_study import training_dir as base_dir
from .training import _move_batch_to_device, set_seed
from .position_head_refit import (write_json as _write_json, save_checkpoint, sha256, tensor_hash,
    RefitConfig, fit_cached, head_from_state, predict, ROOT as POSITION_ROOT)
from .event_posterior_online import IntegratedEventModel
from .oracle_dependency import load_roster_team_map
from .fixed_budget_loss import fixed_budget_loss

ROOT = EXPERIMENT_ROOT / 'spatiotemporal_edge_v1'
SEEDS = (20260715, 20260716, 20260717)


def write_json(path, value):
    def clean(v):
        if isinstance(v, dict): return {k:clean(x) for k,x in v.items()}
        if isinstance(v, (list,tuple)): return [clean(x) for x in v]
        if isinstance(v, (float,np.floating)) and not np.isfinite(v): return None
        return v
    _write_json(path, clean(value))


def run_dir(mode, seed, root=ROOT):
    return root / 'training' / mode / f'seed{seed}'


def config_for(mode, seed, device, root=ROOT, smoke=False, workers=2):
    return FixedBudgetConfig(configuration='partial_l2', output_dir=run_dir(mode, seed, root),
        seed=seed, device=device, training_budget=1 if smoke else 24, num_workers=workers,
        max_train_samples=256 if smoke else None, max_validation_samples=64 if smoke else None)


def require_test(root=ROOT, mode=None):
    path = root / 'selection/spatiotemporal_lock.json'
    if not path.exists():
        raise RuntimeError('Test access forbidden before validation lock')
    lock = json.loads(path.read_text())
    if lock['selected'] == 'base' or (mode is not None and mode not in lock['test_models']):
        raise RuntimeError('No new test evaluation authorized for this model')
    for path, digest in lock['checkpoints'].items():
        if sha256(Path(path)) != digest: raise RuntimeError('Locked checkpoint changed')
    return lock


def provenance():
    files = list(Path(__file__).parent.glob('*.py')) + [FEASIBILITY_ARTIFACT, SAMPLE_PLAN]
    files += list((POSSESSION_GRAPH_ROOT/'metadata').glob('*'))
    for seed in SEEDS:
        files += [base_dir(seed)/'best_guarded_core.pt',base_dir(seed)/'result.json',
                  five_dir('five_f80',seed)/'result.json',
                  POSITION_ROOT/'training'/f'seed{seed}'/'best_position.pt']
    return {str(p): sha256(p) for p in sorted(files)}


def train(mode, seed, device='cpu', root=ROOT, smoke=False, workers=2, resume=True):
    cfg = config_for(mode, seed, device, root, smoke, workers)
    out = cfg.output_dir; out.mkdir(parents=True, exist_ok=True)
    if (out / 'result.json').exists():
        return json.loads((out / 'result.json').read_text())
    set_seed(seed)
    dev = torch.device(device)
    if dev.type == 'cuda':
        torch.cuda.set_device(dev); torch.cuda.reset_peak_memory_stats(dev)
    artifacts = ProtocolArtifacts.load(FEASIBILITY_ARTIFACT)
    model = SpatiotemporalHGT(artifacts, mode).to(dev)
    common_hash = _common_hash(model)
    baseline_result = json.loads((base_dir(seed) / 'result.json').read_text())
    expected = baseline_result['initial_common_sha256']
    if expected != common_hash:
        raise RuntimeError('Baseline initialization hash mismatch')
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    loaders = {s: _loader(s, _loader_config(cfg), artifacts, s == 'train') for s in ('train', 'validation')}
    diagnostic = _move_batch_to_device(next(iter(_loader('validation', _loader_config(cfg), artifacts, False, batch_size=8))), dev)
    # Keep the original training protocol's pre-training diagnostic RNG consumption.
    initial = _gradient_diagnostics(model, diagnostic, artifacts, ALL_TASKS)
    layers = _partial_layer_gradient_diagnostics(model, diagnostic, artifacts, ALL_TASKS)
    guards = [(_guard_metrics(five_dir('five_f80', seed) / 'result.json'), .01),
              (_guard_metrics(base_dir(seed) / 'result.json'), .005)]
    history = []; elapsed = 0.; sources = provenance()
    last = out / 'last.pt'
    if resume and last.exists():
        saved = torch.load(last, map_location='cpu', weights_only=False)
        identity = dict(asdict(cfg)); identity['device'] = None
        old = dict(saved['config']); old['device'] = None
        if old != identity or saved['edge_mode'] != mode or saved['sources'] != sources:
            raise RuntimeError('Resume inputs, code or configuration changed')
        model.load_state_dict(saved['model']); optimizer.load_state_dict(saved['optimizer'])
        history, elapsed = saved['history'], saved['elapsed_seconds']
        initial, layers = saved['initial_gradients'], saved['initial_layer_gradients']
        _restore_rng(saved['rng'], loaders['train'])
    write_json(out / 'config.json', {**asdict(cfg), 'edge_mode': mode, 'early_stopping': False,
        'common_initialization_sha256': common_hash, 'sources': sources, 'guards': guards})
    best = {key: min((r[metric] for r in history if key != 'guarded_core' or r['eligible']), default=float('inf'))
            for key, metric in [('core', 'core_etp_loss'), ('joint', 'joint_active_loss'), ('guarded_core', 'core_etp_loss')]}
    start = time.monotonic()
    for epoch in range(len(history) + 1, cfg.training_budget + 1):
        tick = time.monotonic()
        training = _train_epoch(model, loaders['train'], optimizer, artifacts, cfg, dev)
        with torch.no_grad():
            metrics, _ = evaluate_fixed_budget(model, loaders['validation'], artifacts, ALL_TASKS, dev)
            model(diagnostic)
        if not all(np.isfinite(metrics[k]) for k in ('core_etp_loss','joint_active_loss')):
            raise RuntimeError('Non-finite validation loss')
        row = {'epoch': epoch, 'train': training, 'validation': metrics,
               'core_etp_loss': metrics['core_etp_loss'], 'joint_active_loss': metrics['joint_active_loss'],
               'eligible': all(guarded_core_eligible(metrics, g, margin=m) for g, m in guards),
               'seconds': time.monotonic()-tick, 'gates': model.gate_diagnostics()}
        history.append(row)
        payload = {'architecture': 'spatiotemporal_edge_v1', 'edge_mode': mode, 'config': asdict(cfg),
            'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch,
            'history': history, 'rng': _rng_state(loaders['train']), 'sources': sources,
            'common_initialization_sha256': common_hash, 'initial_gradients': initial,
            'initial_layer_gradients': layers, 'elapsed_seconds': elapsed + time.monotonic()-start}
        for key, metric in [('core', 'core_etp_loss'), ('joint', 'joint_active_loss'), ('guarded_core', 'core_etp_loss')]:
            if (key != 'guarded_core' or row['eligible']) and row[metric] < best[key] - 1e-12:
                best[key] = row[metric]; save_checkpoint(out / f'best_{key}.pt', payload)
        save_checkpoint(last, payload); write_json(out / 'history.json', history)
        print(json.dumps({'mode': mode, 'seed': seed, **row}), flush=True)
    result = {'mode': mode, 'seed': seed, 'completed_epochs': len(history), 'eligible': (out/'best_guarded_core.pt').exists(),
              'common_initialization_sha256': common_hash, 'parameter_count': sum(p.numel() for p in model.parameters()),
              'training_seconds': elapsed + time.monotonic()-start,
              'peak_memory_bytes': torch.cuda.max_memory_allocated(dev) if dev.type == 'cuda' else None,
              'test_accessed': False}
    if result['eligible']:
        state = torch.load(out/'best_guarded_core.pt', map_location='cpu', weights_only=False)
        model.load_state_dict(state['model'])
        with torch.no_grad():
            result['validation'], frame = evaluate_fixed_budget(model, loaders['validation'], artifacts, ALL_TASKS, dev)
        frame.to_parquet(out/'validation_raw.parquet', index=False)
    write_json(out/'result.json', result)
    return result


def load_backbone(mode, seed, device, root=ROOT):
    path = run_dir(mode, seed, root)/'best_guarded_core.pt'
    saved = torch.load(path, map_location='cpu', weights_only=False)
    model = SpatiotemporalHGT(ProtocolArtifacts.load(FEASIBILITY_ARTIFACT), mode)
    model.load_state_dict(saved['model'])
    return model.to(device).requires_grad_(False).eval(), path


@torch.no_grad()
def cache_context(mode, seed, split, device='cpu', root=ROOT):
    if split == 'test': require_test(root, mode)
    out = run_dir(mode, seed, root); target = out/f'{split}_context.pt'
    model, checkpoint = load_backbone(mode, seed, device, root)
    origin = sha256(checkpoint)
    if target.exists():
        cached = torch.load(target, map_location='cpu', weights_only=False)
        if cached['checkpoint_sha256'] != origin: raise RuntimeError('Stale context cache')
        return cached
    artifacts = ProtocolArtifacts.load(FEASIBILITY_ARTIFACT)
    cfg = config_for(mode, seed, device, root)
    if split == 'test':
        from dataclasses import replace
        cfg = replace(cfg, evaluate_test=True, full_test=True)
    loader = _loader(split, _loader_config(cfg), artifacts, False)
    values = {k: [] for k in ('main_context', 'position_true', 'position_mask', 'player_mask',
        'event_true', 'zone_true', 'match_ids', 'current_event_indices', 'base_position_xy')}
    ids = []
    for raw in loader:
        batch = _move_batch_to_device(raw, torch.device(device))
        prediction, contexts = model.forward_with_contexts(batch)
        context = contexts['f80'].detach()
        if float((model.position_head(context).sigmoid() - prediction['position_xy']).abs().max()) >= 1e-6:
            raise RuntimeError('Context/Head cache mismatch')
        t = batch['targets']
        row = {'main_context': context, 'position_true': t['position_xy'], 'position_mask': t['position_mask'],
            'player_mask': t['player_mask'], 'event_true': t['raw_event_10'], 'zone_true': t['zone_20'],
            'match_ids': batch['match_ids'], 'current_event_indices': batch['current_event_indices'],
            'base_position_xy': prediction['position_xy']}
        for key, tensor in row.items(): values[key].append(tensor.detach().cpu())
        ids.extend(raw['sample_ids'])
    cache = {k: torch.cat(v) for k, v in values.items()}
    cache.update(sample_ids=ids, checkpoint_sha256=origin)
    expected = {'train': 34048, 'validation': 7296, 'test': 96854}[split]
    if len(ids) != expected or len(set(ids)) != len(ids): raise RuntimeError('Unexpected sample set')
    save_checkpoint(target, cache)
    return cache


def refit(mode, seed, device='cpu', root=ROOT):
    out = run_dir(mode, seed, root)
    if not json.loads((out/'result.json').read_text())['eligible']: return
    train_cache = cache_context(mode, seed, 'train', device, root)
    val_cache = cache_context(mode, seed, 'validation', device, root)
    model, path = load_backbone(mode, seed, 'cpu', root)
    head = head_from_state(model.state_dict())
    if (predict(head, val_cache['main_context'])-val_cache['base_position_xy']).abs().max() >= 1e-6:
        raise RuntimeError('Refit epoch zero mismatch')
    output = out/'position_refit'
    fit_cached(head, train_cache, val_cache, RefitConfig(seed=seed), output,
        {'backbone': str(path), 'sha256': sha256(path)},
        resume_from=output/'last.pt' if (output/'last.pt').exists() else None)
    evaluate_deployed(mode, seed, 'validation', device, root)


@torch.no_grad()
def evaluate_deployed(mode, seed, split, device='cpu', root=ROOT):
    if split == 'test': require_test(root, mode)
    artifacts = ProtocolArtifacts.load(FEASIBILITY_ARTIFACT)
    if mode == 'base':
        from .event_posterior_online import load_integrated_model
        model = load_integrated_model(seed, device=device)
    else:
        backbone, _ = load_backbone(mode, seed, device, root)
        head = torch.load(run_dir(mode, seed, root)/'position_refit/best_position.pt', map_location='cpu', weights_only=False)
        backbone.position_head.load_state_dict(head['head'])
        model = IntegratedEventModel(backbone, artifacts, load_roster_team_map()).to(device).eval()
    cfg = config_for(mode, seed, device, root)
    if split == 'test':
        from dataclasses import replace
        cfg = replace(cfg, evaluate_test=True, full_test=True)
    loader = _loader(split, _loader_config(cfg), artifacts, False)
    metrics, frame = evaluate_fixed_budget(model, loader, artifacts, ALL_TASKS, torch.device(device))
    if len(frame) != (7296 if split == 'validation' else 96854): raise RuntimeError('Wrong evaluation size')
    out = root/'predictions'/split/mode; out.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(out/f'seed{seed}.parquet', index=False)
    write_json(out/f'seed{seed}.json', metrics)
    return frame
