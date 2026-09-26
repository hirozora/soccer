#!/usr/bin/env python
"""Independent Rotate-based ECA pipeline, with validation-only selection."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

VERSION = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(VERSION / 'src'), str(VERSION.parent / 'benchmark_unified_v1/src')]
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[name] = '1'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
os.environ['PYTHONUNBUFFERED'] = '1'
import torch
torch.set_num_threads(1)
torch.backends.mkldnn.enabled = False
torch.use_deterministic_algorithms(True)

from football_hgt_targets_v4.eca_training import ROOT, SEEDS, sources, train_eca, refit_eca, write_json
from football_hgt_targets_v4.eca_verification import audit, smoke, diagnostics
from football_hgt_targets_v4.eca_reporting import selection, test_locked, report


def verify(root):
    out = root / 'verification'
    out.mkdir(parents=True, exist_ok=True)
    with (out / 'unit_tests.log').open('w') as handle:
        result = subprocess.run([sys.executable, '-m', 'pytest', 'tests/test_eca_transition.py', '-q'],
            cwd=VERSION, env={**os.environ, 'PYTHONPATH': os.pathsep.join(sys.path[:2])},
            stdout=handle, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f'Unit tests failed: {out / "unit_tests.log"}')
    audit(root)
    write_json(out / 'unit_passed.json', {'passed': True, 'sources': sources()})


def require_verified(root):
    for name in ('unit_passed.json', 'smoke.json'):
        path = root / 'verification' / name
        if not path.exists():
            raise RuntimeError(f'Preflight missing: {path}')
        value = json.loads(path.read_text())
        if not value['passed'] or value['sources'] != sources():
            raise RuntimeError('Preflight source hashes are stale')


def matrix(root, stage, tasks, devices):
    pending, active, states, attempts = list(tasks), [], {}, {}
    out = root / 'background'; out.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available() or not devices or len(set(devices)) != len(devices):
        raise RuntimeError('Unique CUDA devices required')
    try:
        while pending or active:
            free = [d for d in devices[:4] if d not in [x['device'] for x in active]]
            while pending and free:
                mode, seed = pending.pop(0)
                device = free.pop(0)
                key = f'{mode}-{seed}'
                attempts[key] = attempts.get(key, 0) + 1
                log = out / f'{stage}-{key}.log'
                handle = log.open('a')
                command = [sys.executable, str(Path(__file__).resolve()), '--stage', stage,
                           '--mode', mode, '--seed', str(seed), '--device', f'cuda:{device}', '--root', str(root)]
                process = subprocess.Popen(command, cwd=VERSION, env=os.environ.copy(), stdout=handle, stderr=subprocess.STDOUT)
                active.append(dict(process=process, handle=handle, device=device, key=key, mode=mode, seed=seed))
                states[key] = {'pid': process.pid, 'status': 'running', 'attempt': attempts[key], 'device': device,
                               'log': str(log), 'command': command, 'started_at': time.time()}
            for item in list(active):
                code = item['process'].poll()
                if code is None:
                    continue
                item['handle'].close(); active.remove(item)
                states[item['key']].update(status='complete' if code == 0 else 'failed', exit_code=code, finished_at=time.time())
                if code and attempts[item['key']] < 2:
                    pending.append((item['mode'], item['seed']))
            write_json(out / f'{stage}_status.json', states)
            if active:
                time.sleep(10)
        if any(v['status'] != 'complete' for v in states.values()):
            raise RuntimeError(f'{stage} failed; downstream stages not started')
    finally:
        for item in active:
            item['process'].terminate()
        for item in active:
            item['process'].wait(); item['handle'].close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', required=True, choices=('verify', 'smoke', 'train', 'refit', 'select', 'test', 'report', 'pipeline'))
    parser.add_argument('--mode', choices=('constant', 'transition'))
    parser.add_argument('--seed', type=int, choices=SEEDS)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--devices', nargs='+', type=int, default=[0, 1, 2, 3])
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--resume', '--resume-from', dest='resume', type=Path)
    args = parser.parse_args()
    root = args.root.resolve(); root.mkdir(parents=True, exist_ok=True)
    if args.seed is not None or args.mode is not None:
        if args.seed is None or args.mode is None or args.stage not in ('train', 'refit'):
            parser.error('Worker requires mode, seed and train/refit stage')
        require_verified(root)
        if args.stage == 'train':
            train_eca(args.mode, args.seed, args.device, root, resume_from=args.resume)
        else:
            refit_eca(args.mode, args.seed, args.device, root)
        return
    if args.resume:
        parser.error('--resume requires a train worker')
    def status(stage, state='running', **extra):
        write_json(root / 'background/pipeline_status.json', {'pid': os.getpid(), 'stage': stage,
                   'status': state, 'updated_at': time.time(), **extra})
    try:
        if args.stage in ('verify', 'pipeline'):
            status('verify'); verify(root)
        if args.stage in ('smoke', 'pipeline'):
            status('smoke'); smoke(root, args.device)
        tasks = [(m, s) for m in ('constant', 'transition') for s in SEEDS]
        if args.stage in ('train', 'pipeline'):
            require_verified(root)
            if not (root / 'selection/eca_lock.json').exists():
                status('train'); matrix(root, 'train', tasks, args.devices)
        if args.stage in ('refit', 'pipeline'):
            status('refit'); matrix(root, 'refit', tasks, args.devices)
        if args.stage in ('select', 'pipeline'):
            status('select'); selection(root)
        if args.stage in ('test', 'pipeline'):
            status('test'); test_locked(root, args.device)
        if args.stage in ('report', 'pipeline'):
            status('diagnostics')
            for mode in ('base', 'constant', 'transition'):
                for seed in SEEDS:
                    diagnostics(mode, seed, args.device, root)
            report(root)
        status(args.stage, 'completed')
    except BaseException as error:
        status(args.stage, 'failed', error=repr(error))
        raise


if __name__ == '__main__':
    main()
