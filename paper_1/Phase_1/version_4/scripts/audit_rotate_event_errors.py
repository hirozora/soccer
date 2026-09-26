"""Descriptive, validation-only error audit of Original and Rotate predictions."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

V4 = Path(__file__).resolve().parents[1]
PHASE = V4.parent
SOURCE = V4 / 'experiments/coverage_history_v1'
GRAPHS = PHASE / 'data/whyscout/processed/heterogeneous_graphs/semantic_v3_possession'
OUT = V4 / 'experiments/rotate_event_error_audit_v1'
NAMES = ['Duel', 'Foul', 'Free Kick', 'Goalkeeper leaving line', 'Interruption',
         'Offside', 'Others on the ball', 'Pass', 'Save attempt', 'Shot']
SEEDS = [20260715, 20260716, 20260717]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def group_stats(frame, mask, name, model, seed):
    wrong = frame.event_true != frame.event_pred
    forward = frame.event_true.isin([0, 6]) & (frame.event_pred == 7)
    reverse = (frame.event_true == 7) & frame.event_pred.isin([0, 6])
    n = int(mask.sum())
    def rate(count, denominator):
        return float(count / denominator) if denominator else None
    return dict(model=model, seed=seed, group=name, samples=n,
        sample_share=rate(n, len(frame)), errors=int((mask & wrong).sum()),
        error_rate=rate((mask & wrong).sum(), n),
        forward_errors=int((mask & forward).sum()), reverse_errors=int((mask & reverse).sum()),
        forward_rate=rate((mask & forward).sum(), n),
        forward_share_of_all_errors=rate((mask & forward).sum(), wrong.sum()),
        forward_share_of_all_forward=rate((mask & forward).sum(), forward.sum()),
        true_duel_others=int((mask & frame.event_true.isin([0, 6])).sum()),
        forward_given_true_duel_others=rate((mask & forward).sum(), (mask & frame.event_true.isin([0, 6])).sum()),
        true_pass=int((mask & (frame.event_true == 7)).sum()),
        reverse_given_true_pass=rate((mask & reverse).sum(), (mask & (frame.event_true == 7)).sum()))


def main():
    torch.set_num_threads(1)
    OUT.mkdir(parents=True, exist_ok=True)
    paths = {(m, s): SOURCE / f'predictions/validation/{m}/seed{s}.parquet'
             for m in ('original', 'rotate') for s in SEEDS}
    frames = {k: pd.read_parquet(p).sort_values('sample_id').reset_index(drop=True) for k, p in paths.items()}
    reference = frames['original', SEEDS[0]]
    keys = ['sample_id', 'match_id', 'current_event_index', 'event_true']
    assert len(reference) == 96891 and reference.sample_id.is_unique
    assert all(f[keys].equals(reference[keys]) for f in frames.values())
    index = pd.read_csv(GRAPHS / 'metadata/match_index.csv').set_index('match_id')
    vocab_path = GRAPHS / 'metadata/vocabularies.json'
    vocab = json.loads(vocab_path.read_text())['values']['control_state']
    context = reference[keys].copy()
    graph_hashes = {}
    for match, rows in context.groupby('match_id').groups.items():
        path = GRAPHS / index.loc[match, 'graph_path']
        graph_hashes[str(path)] = digest(path)
        event = torch.load(path, map_location='cpu', weights_only=False)['node_stores']['event']
        anchor = context.loc[rows, 'current_event_index'].to_numpy(int)
        assert np.array_equal(event['event_type_index'][anchor + 1].numpy(), context.loc[rows, 'event_true'])
        context.loc[rows, 'anchor_type'] = event['event_type_index'][anchor].numpy()
        context.loc[rows, 'anchor_control'] = event['control_state_after_index'][anchor].numpy()
    classes, pairs, groups, totals = [], [], [], []
    for (model, seed), frame in frames.items():
        y, pred = frame.event_true.to_numpy(int), frame.event_pred.to_numpy(int)
        matrix = confusion_matrix(y, pred, labels=range(10))
        assert matrix.sum() == len(frame)
        precision, recall, f1, support = precision_recall_fscore_support(y, pred, labels=range(10), zero_division=0)
        errors = int((y != pred).sum())
        totals.append(dict(model=model, seed=seed, accuracy=float((y == pred).mean()),
                           macro_f1=float(f1.mean()), errors=errors))
        for i, name in enumerate(NAMES):
            classes.append(dict(model=model, seed=seed, event=name, support=int(support[i]),
                precision=precision[i], recall=recall[i], f1=f1[i], predicted_count=int(matrix[:, i].sum())))
            for j in range(10):
                if i != j:
                    pairs.append(dict(model=model, seed=seed, true_event=name, predicted_event=NAMES[j],
                        count=int(matrix[i, j]), share_of_errors=matrix[i, j] / errors,
                        rate_given_true=matrix[i, j] / max(1, support[i])))
        masks = {'ALL': np.ones(len(frame), dtype=bool),
                 'Pass + CONTROL': (context.anchor_type == 7) & (context.anchor_control == vocab.index('CONTROL'))}
        masks['Other anchors'] = ~masks['Pass + CONTROL']
        for (typ, control), rows in context.groupby(['anchor_type', 'anchor_control']).groups.items():
            name = f'{NAMES[int(typ)]} / {vocab[int(control)]}'
            masks[name] = context.index.isin(rows)
        for name, mask in masks.items():
            groups.append(group_stats(frame, mask, name, model, seed))
    outputs = {'overall': pd.DataFrame(totals), 'per_class': pd.DataFrame(classes),
               'confusions': pd.DataFrame(pairs), 'anchor_groups': pd.DataFrame(groups)}
    grouping = {'overall': ['model'], 'per_class': ['model', 'event'],
                'confusions': ['model', 'true_event', 'predicted_event'], 'anchor_groups': ['model', 'group']}
    means = {}
    for name, df in outputs.items():
        df.to_csv(OUT / f'{name}_per_seed.csv', index=False)
        numbers = [c for c in df.select_dtypes(include='number') if c != 'seed']
        mean = df.groupby(grouping[name], sort=False)[numbers].mean().reset_index()
        means[name] = mean
        mean.to_csv(OUT / f'{name}_mean.csv', index=False)
    top = means['confusions'].query("model == 'rotate'").sort_values('count', ascending=False).head(10)
    focal = means['anchor_groups'].query("group in ['ALL', 'Pass + CONTROL', 'Other anchors']")
    summary = {'validation_samples': len(reference), 'matches': int(reference.match_id.nunique()),
        'seeds': SEEDS, 'overall': means['overall'].to_dict('records'),
        'top_rotate_confusions': top.to_dict('records'), 'focus_groups': focal.to_dict('records'),
        'test_accessed': False, 'training_performed': False,
        'interpretation': 'Counts are means per seed; support counts represent unique samples, not three independent samples. Descriptive only.'}
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    (OUT / 'manifest.json').write_text(json.dumps({'predictions': {str(p): digest(p) for p in paths.values()},
        'graphs': graph_hashes, 'vocab_sha256': digest(vocab_path), 'script_sha256': digest(Path(__file__))}, indent=2) + '\n')
    lines = ['# Rotate Event Error Audit', '', 'Full validation only: 96,891 unique samples, 57 matches, three seeds.',
        'Original and Rotate use exactly the same samples. No training or test access.',
        'Counts are per-seed means; class recall is correctness conditional on the true class.', '',
        '| Model | Accuracy | Macro-F1 | Errors |', '|---|---:|---:|---:|']
    for r in summary['overall']:
        lines.append(f"| {r['model']} | {r['accuracy']:.4%} | {r['macro_f1']:.4f} | {r['errors']:.1f} |")
    lines += ['', 'See per_class_mean.csv, confusions_mean.csv and anchor_groups_mean.csv.',
        'Focus: Duel/Others -> Pass, reverse Pass -> Duel/Others, and anchor Pass + CONTROL.',
        'The old 7,296-sample audit is not directly comparable by raw counts.',
        'Error concentration does not establish a causal attention/hub failure.']
    (OUT / 'README.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
