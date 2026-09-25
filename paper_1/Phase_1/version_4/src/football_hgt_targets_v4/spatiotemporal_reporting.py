"""Match-only paired inference and validation-only model selection."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from .spatiotemporal_training import ROOT, SEEDS, run_dir, require_test, write_json
from .position_head_refit import sha256, ROOT as POSITION_ROOT
from .partial_sharing_study import training_dir as base_dir, test_dir as base_test_dir
from .team_candidate_prior_study import cache_path as team_cache
from .team_candidate_prior import prediction_frame
from .five_task_reporting import _match_statistics, _to_metrics, METRICS
from .metrics import compute_metrics
from .actor_training import compute_actor_metrics

THRESHOLDS = {'event_macro_f1': .005, 'time_mae_seconds': .01, 'position_distance_mae_m': .25}
DIRECTION = {'event_macro_f1': 1, 'time_mae_seconds': -1, 'position_distance_mae_m': -1}


def frame_path(mode, seed, split, root=ROOT):
    if split == 'test': require_test(root, mode)
    return root/'predictions'/split/mode/f'seed{seed}.parquet'


def align(left, right):
    left = left.sort_values('sample_id').reset_index(drop=True)
    right = right.sort_values('sample_id').reset_index(drop=True)
    keys = ['sample_id', 'match_id', 'current_event_index', 'event_true', 'time_true', 'time_mask',
            'position_true_x', 'position_true_y', 'position_mask', 'player_mask', 'team_true']
    for key in keys:
        if not np.array_equal(left[key].to_numpy(), right[key].to_numpy()):
            raise RuntimeError(f'Prediction alignment mismatch: {key}')
    if left.sample_id.duplicated().any() or right.sample_id.duplicated().any():
        raise RuntimeError('Duplicate sample ID')
    return left, right


def core_loss(frame):
    # Same postprocessing metric for reused Base and new backbones; all valid samples.
    columns = [c for c in frame if c.startswith('event_probability_')]
    columns.sort(key=lambda c: int(c.split('_')[2]))
    probs = frame[columns].to_numpy()
    event = -np.log(np.maximum(probs[np.arange(len(frame)), frame.event_true.to_numpy(int)], 1e-30)).mean()/np.log(10)
    t = frame[frame.time_mask]
    error = np.abs(t.time_pred.to_numpy()-t.time_true.to_numpy())/60
    beta = 1/60
    time = np.where(error < beta, .5*error**2/beta, error-.5*beta).mean()
    p = frame[frame.position_mask]
    xy = p[['position_pred_x','position_pred_y']].to_numpy()-p[['position_true_x','position_true_y']].to_numpy()
    error = np.abs(xy)
    position = np.where(error < 1, .5*error**2, error-.5).mean()
    return float((.2*event+time+position)/3)


def reuse_base(seed, split='validation', root=ROOT):
    target = frame_path('base', seed, split, root)
    if target.exists(): return
    original = (base_dir(seed)/'validation_predictions_guarded_core.parquet' if split == 'validation'
                else base_test_dir(seed)/'test_predictions.parquet')
    position = (POSITION_ROOT/'training'/f'seed{seed}'/'validation_refit.parquet' if split == 'validation'
                else POSITION_ROOT/'test'/f'seed{seed}'/'test_refit.parquet')
    frame = pd.read_parquet(original).sort_values('sample_id').reset_index(drop=True)
    refit = pd.read_parquet(position).sort_values('sample_id').reset_index(drop=True)
    for col in ['sample_id','match_id','current_event_index','position_true_x','position_true_y','position_mask','player_mask']:
        if not np.array_equal(frame[col], refit[col]): raise RuntimeError(f'Base Refit mismatch {col}')
    for col in ['position_pred_x','position_pred_y']: frame[col] = refit[col]
    from .metrics import position_to_zone
    frame['zone_pred'] = position_to_zone(torch.tensor(frame[['position_pred_x','position_pred_y']].to_numpy(),dtype=torch.float32)).numpy()
    cache = torch.load(team_cache(seed, split), map_location='cpu', weights_only=False)
    tc = prediction_frame(cache, 'soft', lambda_value=1.5).sort_values('sample_id').reset_index(drop=True)
    for col in ['sample_id', 'match_id', 'current_event_index', 'player_mask', 'team_true']:
        if not np.array_equal(frame[col], tc[col]): raise RuntimeError(f'Base TC mismatch {col}')
    frame['player_rank'] = tc.player_rank
    target.parent.mkdir(parents=True, exist_ok=True); frame.to_parquet(target, index=False)
    write_json(target.with_suffix('.sources.json'), {str(p): sha256(p) for p in [original, position, team_cache(seed, split)]})


def compare(reference, candidate, iterations=10000):
    pairs = [align(a, b) for a, b in zip(reference, candidate)]
    if len(pairs) != 3: raise ValueError('Three paired seeds required')
    for a, _ in pairs[1:]: align(pairs[0][0], a)
    matches = sorted(pairs[0][0].match_id.unique().tolist())
    a = np.stack([_match_statistics(x, matches) for x, _ in pairs])
    b = np.stack([_match_statistics(y, matches) for _, y in pairs])
    observed_a, observed_b = _to_metrics(a.sum(1)), _to_metrics(b.sum(1))
    draws = np.random.default_rng(20260715).multinomial(len(matches), np.full(len(matches), 1/len(matches)), iterations)
    bootstrap = {key: [] for key in METRICS}
    for start in range(0, iterations, 250):
        d = draws[start:start+250]
        ma = _to_metrics(np.einsum('bm,smw->bsw', d, a))
        mb = _to_metrics(np.einsum('bm,smw->bsw', d, b))
        for key in METRICS: bootstrap[key].extend((mb[key]-ma[key]).mean(1).tolist())
    result = {}
    for key in METRICS:
        delta = observed_b[key]-observed_a[key]
        low, high = np.quantile(bootstrap[key], [.025,.975])
        result[key] = {'difference': float(delta.mean()), 'per_seed': delta.tolist(),
                       'ci_low': float(low), 'ci_high': float(high)}
    return result


def effective(comparison):
    return [k for k, threshold in THRESHOLDS.items() if
        comparison[k]['difference']*DIRECTION[k] >= threshold and
        sum(v*DIRECTION[k] > 0 for v in comparison[k]['per_seed']) >= 2 and
        (comparison[k]['ci_low'] > 0 if DIRECTION[k] == 1 else comparison[k]['ci_high'] < 0)]


def guards(comparison):
    d = {k: v['difference'] for k, v in comparison.items()}
    return (d['event_accuracy'] >= -.01 and d['event_macro_f1'] > -.02 and
        d['time_mae_seconds'] < .05 and d['position_distance_mae_m'] < .5 and
        d['team_accuracy'] >= -.005 and d['player_top1'] >= -.005)


def choose(comparisons, losses, complete):
    eligible = ['base']; evidence = {}
    for mode in ('constant', 'conditioned'):
        if mode not in complete: continue
        vs_base = comparisons[f'{mode}-base']
        tasks = set(effective(vs_base))
        if mode == 'conditioned':
            tasks &= set(effective(comparisons['conditioned-constant'])) if 'constant' in complete else set()
        evidence[mode] = {'effective_tasks': sorted(tasks), 'guards_pass': guards(vs_base)}
        if tasks and guards(vs_base): eligible.append(mode)
    minimum = min(losses[k] for k in eligible)
    selected = next(k for k in ('base','constant','conditioned') if k in eligible and losses[k] < minimum+1e-4)
    return selected, eligible, evidence


def selection(root=ROOT):
    lockpath = root/'selection/spatiotemporal_lock.json'
    if lockpath.exists(): return json.loads(lockpath.read_text())
    complete = ['base']
    for mode in ('constant','conditioned'):
        results = [json.loads((run_dir(mode,s,root)/'result.json').read_text()) for s in SEEDS]
        if any(r['completed_epochs'] != 24 for r in results): raise RuntimeError('Training incomplete')
        if all(r['eligible'] for r in results): complete.append(mode)
    for seed in SEEDS: reuse_base(seed, root=root)
    frames = {m: [pd.read_parquet(frame_path(m,s,'validation',root)) for s in SEEDS] for m in complete}
    comparisons = {}
    for candidate, reference in [('constant','base'),('conditioned','constant'),('conditioned','base')]:
        if candidate in complete and reference in complete:
            comparisons[f'{candidate}-{reference}'] = compare(frames[reference], frames[candidate])
    losses = {m: float(np.mean([core_loss(f) for f in v])) for m, v in frames.items()}
    selected, eligible, evidence = choose(comparisons, losses, complete)
    test_models = [] if selected == 'base' else (['base','constant','conditioned'] if selected == 'conditioned' else ['base','constant'])
    paths = [run_dir(m,s,root)/name for m in test_models if m != 'base' for s in SEEDS
             for name in ('best_guarded_core.pt', 'position_refit/best_position.pt')]
    lock = {'selection_split': 'validation', 'selected': selected, 'eligible': eligible, 'complete': complete,
        'comparisons': comparisons, 'evidence': evidence, 'core_losses': losses, 'test_models': test_models,
        'checkpoints': {str(p):sha256(p) for p in paths}, 'test_accessed': False,
        'bootstrap_unit': 'match', 'bootstrap_iterations': 10000, 'seed_resampling': False}
    write_json(lockpath, lock)
    return lock


def report(root=ROOT):
    lock = json.loads((root/'selection/spatiotemporal_lock.json').read_text())
    rows = []
    raw_rows = []
    for split, modes in [('validation',lock['complete']), ('test',lock['test_models'])]:
        for mode in modes:
            for seed in SEEDS:
                f = pd.read_parquet(frame_path(mode,seed,split,root))
                metrics = compute_metrics(f)
                metrics['team'] = compute_actor_metrics(f,'team',0.)['team']
                metrics['player'] = compute_actor_metrics(f,'player',0.)['player']
                path = root/'reports'/split/mode/f'seed{seed}.json'; write_json(path, metrics)
                stats = _to_metrics(_match_statistics(f, sorted(f.match_id.unique())).sum(0))
                rows.append({'split':split,'mode':mode,'seed':seed,**{k:float(v) for k,v in stats.items()},
                             'position_median_m':metrics['position']['distance_median_m'], 'core_loss':core_loss(f)})
                if split == 'validation':
                    raw_path = (base_dir(seed)/'validation_predictions_guarded_core.parquet' if mode=='base'
                                else run_dir(mode,seed,root)/'validation_raw.parquet')
                    raw_frame = pd.read_parquet(raw_path)
                    align(raw_frame,f)
                    raw_stats = _to_metrics(_match_statistics(raw_frame,sorted(raw_frame.match_id.unique())).sum(0))
                    raw_rows.append({'mode':mode,'seed':seed,**{k:float(v) for k,v in raw_stats.items()}})
    output = root/'reports'; output.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(rows); table.to_csv(output/'metrics.csv',index=False)
    pd.DataFrame(raw_rows).to_csv(output/'raw_validation_metrics.csv',index=False)
    table.groupby(['split','mode']).agg({c:['mean','std'] for c in METRICS}).to_csv(output/'summary.csv')
    if lock['test_models']:
        frames = {m:[pd.read_parquet(frame_path(m,s,'test',root)) for s in SEEDS] for m in lock['test_models']}
        comparisons = {f'{m}-{r}':compare(frames[r],frames[m]) for m,r in
            [('constant','base'),('conditioned','base'),('conditioned','constant')] if m in frames and r in frames}
        write_json(output/'test_bootstrap.json', comparisons)
    lines = ['# Continuous Spatiotemporal Edge Experiment', '', f"Validation-locked method: **{lock['selected']}**.",
        '', 'Selection uses post-Refit Position and TC-SoftPred (lambda=1.5) Player outputs.',
        'Bootstrap resamples matches only, with the same draws across configurations and seeds.',
        'See metrics.csv, summary.csv, per-seed confusion matrices and the validation lock for numerical evidence.',
        '', 'No test-based reselection. Failure is specific to adjacent-event scalar gating, not all spatiotemporal propagation.',
        'Next research stage: cross-match historical coverage and causal data interfaces. No further gate expansion.']
    (output/'README.md').write_text('\n'.join(lines)+'\n')
