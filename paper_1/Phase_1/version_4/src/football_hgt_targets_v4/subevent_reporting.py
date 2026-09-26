"""Validation-only subevent auxiliary selection, paired match inference and confusion audit."""
import json

import numpy as np
import pandas as pd
import torch

from .subevent_training import (ROOT, ROTATE_ROOT, SEEDS, run_dir, frame_path, require_test,
                          evaluate_deployed, write_json, sources)
from .position_head_refit import sha256
from .spatiotemporal_reporting import compare, core_loss, guards, align
from .five_task_reporting import _match_statistics, _to_metrics
from .metrics import compute_metrics
from .constants import POSSESSION_GRAPH_ROOT


def event_effective(comparison):
    value = comparison['event_macro_f1']
    return (value['difference'] >= .005 and value['ci_low'] > 0
            and sum(d > 0 for d in value['per_seed']) >= 2)


def choose(comparisons, losses, complete):
    eligible = ['base']
    evidence = {}
    for mode in ('coarse', 'fine'):
        if mode not in complete:
            continue
        c = comparisons[f'{mode}-base']
        allowed = event_effective(c) and guards(c)
        if mode == 'fine':
            other = comparisons.get('fine-coarse')
            allowed = (allowed and other is not None and event_effective(other)
                       and other['event_accuracy']['difference'] >= -.01)
        evidence[mode] = {'effective_vs_base': event_effective(c), 'guards_vs_base': guards(c),
                          'eligible': bool(allowed)}
        if allowed:
            eligible.append(mode)
    minimum = min(losses[m] for m in eligible)
    selected = next(m for m in ('base', 'coarse', 'fine')
                    if m in eligible and losses[m] < minimum + 1e-4)
    return selected, eligible, evidence


def reuse_base(seed, split, root=ROOT):
    target = frame_path('base', seed, split, root)
    source = ROTATE_ROOT / 'predictions' / split / 'rotate' / f'seed{seed}.parquet'
    if target.exists():
        if sha256(target) != sha256(source):
            raise RuntimeError('Reused Rotate predictions changed')
    else:
        import shutil
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    write_json(target.with_suffix('.source.json'), {'path': str(source), 'sha256': sha256(source)})


def selection(root=ROOT):
    path = root / 'selection/subevent_auxiliary_lock.json'
    if path.exists():
        return json.loads(path.read_text())
    complete = ['base']
    for mode in ('coarse', 'fine'):
        results = [json.loads((run_dir(mode, s, root) / 'result.json').read_text()) for s in SEEDS]
        if any(r['epochs'] != 24 or r['steps'] != 3192 for r in results):
            raise RuntimeError('Incomplete subevent auxiliary budget')
        if all(r['eligible'] for r in results):
            complete.append(mode)
    for seed in SEEDS:
        reuse_base(seed, 'validation', root)
    frames = {m: [pd.read_parquet(frame_path(m, s, root=root)) for s in SEEDS] for m in complete}
    comparisons = {f'{a}-{b}': compare(frames[b], frames[a]) for a, b in
                   (('coarse', 'base'), ('fine', 'coarse'), ('fine', 'base'))
                   if a in complete and b in complete}
    losses = {m: float(np.mean([core_loss(f) for f in fs])) for m, fs in frames.items()}
    selected, eligible, evidence = choose(comparisons, losses, complete)
    test_models = [] if selected == 'base' else (['base', 'coarse', 'fine']
                                               if selected == 'fine' else ['base', 'coarse'])
    paths = [run_dir(m, s, root) / p for m in complete if m != 'base' for s in SEEDS
             for p in ('best_guarded_core.pt', 'position_refit/best_position.pt')]
    paths += [ROTATE_ROOT / 'coverage/rotate' / f'seed{s}' / p for s in SEEDS
              for p in ('best_guarded_core.pt', 'position_refit/best_position.pt')]
    lock = {'selected': selected, 'eligible': eligible, 'complete': complete,
            'comparisons': comparisons, 'core_losses': losses, 'evidence': evidence,
            'checkpoints': {str(p): sha256(p) for p in paths}, 'sources': sources(),
            'test_models': test_models, 'test_accessed': False, 'selection_split': 'validation',
            'bootstrap_unit': 'match', 'bootstrap_iterations': 10000,
            'validation_samples': 96891, 'existing_split_confirmation': True}
    write_json(path, lock)
    return lock


def test_locked(root=ROOT, device='cuda:0'):
    path = root / 'selection/subevent_auxiliary_lock.json'
    if not path.exists():
        raise RuntimeError('Validation lock required')
    lock = json.loads(path.read_text())
    if not lock['test_models']:
        write_json(root / 'test_status.json', {'status': 'skipped', 'reason': 'Base retained'})
        return
    require_test(root)
    for mode in lock['test_models']:
        for seed in SEEDS:
            if mode == 'base':
                reuse_base(seed, 'test', root)
            else:
                evaluate_deployed(mode, seed, 'test', device, root)
    write_json(root / 'test_status.json', {'status': 'completed', 'locked': lock['selected']})


def anchor_groups(frame):
    index = pd.read_csv(POSSESSION_GRAPH_ROOT / 'metadata/match_index.csv').set_index('match_id')
    result = np.zeros(len(frame), dtype=bool)
    for match, rows in frame.groupby('match_id').groups.items():
        event = torch.load(POSSESSION_GRAPH_ROOT / index.loc[match, 'graph_path'],
                           map_location='cpu', weights_only=False)['node_stores']['event']
        anchors = frame.loc[rows, 'current_event_index'].to_numpy(int)
        result[rows] = ((event['event_type_index'][anchors] == 7)
                        & (event['control_state_after_index'][anchors] == 2)).numpy()
    return result


def report(root=ROOT):
    lock = json.loads((root / 'selection/subevent_auxiliary_lock.json').read_text())
    rows, raw_rows, groups, auxiliary_rows = [], [], [], []
    output = root / 'reports'
    output.mkdir(parents=True, exist_ok=True)
    for split, modes in (('validation', lock['complete']), ('test', lock['test_models'])):
        frames = {}
        focal = None
        for mode in modes:
            frames[mode] = []
            for seed in SEEDS:
                f = pd.read_parquet(frame_path(mode, seed, split, root)).sort_values('sample_id').reset_index(drop=True)
                if focal is None:
                    focal = anchor_groups(f)
                    reference = f
                align(reference, f)
                frames[mode].append(f)
                metrics = compute_metrics(f)
                write_json(output / split / mode / f'seed{seed}.json', metrics)
                statistics = _to_metrics(_match_statistics(f, sorted(f.match_id.unique())).sum(0))
                rows.append({'split': split, 'mode': mode, 'seed': seed,
                             **{k: float(v) for k, v in statistics.items()},
                             'position_median_m': metrics['position']['distance_median_m'], 'core_loss': core_loss(f)})
                forward = f.event_true.isin([0, 6]) & (f.event_pred == 7)
                reverse = (f.event_true == 7) & f.event_pred.isin([0, 6])
                errors = f.event_true != f.event_pred
                for label, mask in (('all', np.ones(len(f), bool)), ('Pass+CONTROL', focal), ('other', ~focal)):
                    groups.append({'split': split, 'mode': mode, 'seed': seed, 'group': label,
                        'samples': int(mask.sum()), 'sample_share': float(mask.mean()),
                        'error_rate': float(errors[mask].mean()), 'forward_errors': int(forward[mask].sum()),
                        'reverse_errors': int(reverse[mask].sum()),
                        'forward_given_duel_others': float(forward[mask].sum() / max(1, f.event_true[mask].isin([0, 6]).sum()))})
                if split == 'validation':
                    directory = (ROTATE_ROOT / 'coverage/rotate' / f'seed{seed}' if mode == 'base' else run_dir(mode, seed, root))
                    state = torch.load(directory / 'best_guarded_core.pt', map_location='cpu', weights_only=False)
                    raw = pd.read_parquet(directory / f"validation_epoch{state['epoch']:02d}.parquet")
                    align(f, raw)
                    stats = _to_metrics(_match_statistics(raw, sorted(raw.match_id.unique())).sum(0))
                    raw_rows.append({'mode': mode, 'seed': seed, 'epoch': state['epoch'], **{k: float(v) for k, v in stats.items()}})
                    if mode != 'base':
                        history = json.loads((directory / 'history.json').read_text())
                        for row in history:
                            if row['validation'] is not None:
                                aux = row['validation']['auxiliary']
                                auxiliary_rows.append({'mode': mode, 'seed': seed, 'epoch': row['epoch'],
                                    'selected': row['epoch'] == state['epoch'],
                                    **{k: v for k, v in aux.items() if k != 'subtype_confusion'}})
                        write_json(output / 'auxiliary' / mode / f'seed{seed}.json',
                            history[state['epoch'] - 1]['validation']['auxiliary'])
        if split == 'test' and modes:
            write_json(output / 'test_bootstrap.json', {f'{a}-{b}': compare(frames[b], frames[a])
                for a, b in (('coarse', 'base'), ('fine', 'coarse'), ('fine', 'base'))
                if a in frames and b in frames})
    table = pd.DataFrame(rows)
    table.to_csv(output / 'metrics.csv', index=False)
    table.groupby(['split', 'mode']).agg({c: ['mean', 'std'] for c in table.select_dtypes('number') if c != 'seed'}).to_csv(output / 'summary.csv')
    pd.DataFrame(raw_rows).to_csv(output / 'raw_validation.csv', index=False)
    pd.DataFrame(groups).to_csv(output / 'anchor_confusions.csv', index=False)
    pd.DataFrame(auxiliary_rows).to_csv(output / 'auxiliary_validation.csv', index=False)
    (output / 'README.md').write_text(
        '# Rotate Next-Subevent Auxiliary Supervision\n\n'
        f"Validation-locked method: **{lock['selected']}**.\n\n"
        'Paired bootstrap resamples matches, sharing draws across configurations and seeds.\n'
        'All selection outputs include backbone-specific Position Refit and fixed TC lambda=1.5.\n'
        'This is confirmation on a previously examined split, not a new blind test.\n'
        'Coarse improvements alone do not establish the value of fine subtype supervision.\n'
        'Overall improvements do not establish the target confusion mechanism without a matching confusion reduction.\n'
        'Auxiliary head removed at inference. No hierarchy aggregation or automatic follow-up experiments.\n')
