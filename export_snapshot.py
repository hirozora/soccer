"""Export a bounded, allowlisted research snapshot without touching source runs."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
from datetime import datetime, timezone


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    args = parser.parse_args()
    destination = Path(__file__).resolve().parent
    phase = args.source / 'paper_1/Phase_1'
    copied, skipped = [], []

    def copy(path):
        relative = path.relative_to(args.source)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        copied.append({'path': str(relative), 'bytes': target.stat().st_size,
                       'sha256': hashlib.sha256(target.read_bytes()).hexdigest()})

    # Source, tests and small reports only. No raw data, caches or machine logs.
    roots = [phase / part for part in ('src', 'scripts', 'tests', 'version_1',
             'version_2', 'version_3', 'version_4', 'benchmark_unified_v1')]
    forbidden = {'__pycache__', '.pytest_cache', 'background', 'logs', 'raw',
                 'history_cache', 'cache', 'caches', 'predictions'}
    for root in roots:
        for path in sorted(root.rglob('*')):
            if not path.is_file() or path.is_symlink():
                continue
            parts = path.relative_to(root).parts
            if any(p in forbidden for p in parts):
                continue
            code = path.suffix in {'.py', '.sh', '.toml'}
            document = path.suffix.lower() in {'.md', '.json', '.csv', '.png', '.svg', '.pdf'}
            document |= path.name.lower() in {'readme', 'requirements.txt'}
            if not (code or document):
                continue
            if path.stat().st_size > 2_000_000:
                skipped.append(str(path.relative_to(args.source)))
                continue
            copy(path)
    for path in sorted(phase.iterdir()):
        if path.is_file() and path.suffix in {'.md', '.txt'}:
            copy(path)
    for path in (phase / 'data/whyscout').glob('*.md'):
        copy(path)
    for path in (phase / 'data/whyscout/processed').rglob('metadata/*.json'):
        if path.stat().st_size < 2_000_000:
            copy(path)
    for seed in (20260715, 20260716, 20260717):
        copy(phase / f'version_4/experiments/layerwise_partial_sharing_v1/training/partial_l2/seed{seed}/best_guarded_core.pt')
        copy(phase / f'version_4/experiments/position_head_refit_v1/training/seed{seed}/best_position.pt')
    for path in (phase / 'benchmark_unified_v1/artifacts').rglob('protocol.pt'):
        copy(path)
    status = phase / 'version_4/experiments/coverage_history_v1/background/pipeline_status.json'
    manifest = {'snapshot_utc': datetime.now(timezone.utc).isoformat(),
                'ongoing_coverage_history': json.loads(status.read_text()),
                'files': copied, 'oversized_reports_excluded': skipped,
                'exclusions': ['raw match/event data', 'graph tensors', 'feature caches',
                               'per-sample predictions', 'most checkpoints', 'logs', 'credentials']}
    (destination / 'SNAPSHOT_MANIFEST.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'files': len(copied), 'bytes': sum(x['bytes'] for x in copied),
                      'oversized_reports': len(skipped)}, indent=2))


if __name__ == '__main__':
    main()
