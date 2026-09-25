#!/usr/bin/env python
"""Independent, resumable continuous spatiotemporal edge experiment."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

VERSION = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(VERSION/'src'), str(VERSION.parent/'benchmark_unified_v1/src')]
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import torch
torch.set_num_threads(1)
torch.backends.mkldnn.enabled = False
from football_hgt_targets_v4.spatiotemporal_training import (
    ROOT, SEEDS, train, refit, run_dir, require_test, evaluate_deployed)
from football_hgt_targets_v4.spatiotemporal_reporting import selection, report, reuse_base
from football_hgt_targets_v4.position_head_refit import write_json, sha256


def verify(root):
    out = root/'verification'; out.mkdir(parents=True, exist_ok=True)
    command = [sys.executable,'-m','pytest','tests/test_spatiotemporal_edge.py', '-q']
    with (out/'unit_tests.log').open('w') as log:
        completed = subprocess.run(command, cwd=VERSION, stdout=log, stderr=subprocess.STDOUT)
    if completed.returncode: raise RuntimeError(f'Unit tests failed: {out}')
    write_json(out/'tests.json', {'passed':True,'command':command})


def smoke(root, device):
    from football_hgt_targets_v4.spatiotemporal_verify import verify_device
    verify_device(root, device)
    for mode in ('constant','conditioned'):
        train(mode, SEEDS[0], device, root/'smoke', smoke=True, workers=0, resume=False)
    write_json(root/'verification/smoke.json', {'passed':True,'device':device})


def matrix(root, stage, devices, slots):
    logroot = root/'background'; logroot.mkdir(parents=True, exist_ok=True)
    pending = [(mode, seed) for mode in ('constant','conditioned') for seed in SEEDS]
    active = []; records = {}; attempts = {}
    capacity = [f'cuda:{i}' for _ in range(slots) for i in devices]
    while pending or active:
        used = [j['device'] for j in active]
        free = capacity.copy()
        for device in used:
            if device in free: free.remove(device)
        while pending and free:
            mode, seed = pending.pop(0); key = f'{mode}-{seed}'
            attempts[key] = attempts.get(key,0)+1
            device = free.pop(0)
            command = [sys.executable, str(Path(__file__).resolve()), '--stage',stage,'--mode',mode,
                '--seed',str(seed),'--device',device,'--root',str(root)]
            path = logroot/f'{stage}-{key}.log'; handle = path.open('a')
            process = subprocess.Popen(command,cwd=VERSION,stdout=handle,stderr=subprocess.STDOUT,
                env={**os.environ,'PYTHONUNBUFFERED':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'})
            job = {'process':process,'handle':handle,'mode':mode,'seed':seed,'device':device,'key':key}
            active.append(job)
            records[key] = {'pid':process.pid,'device':device,'status':'running','attempt':attempts[key],'log':str(path)}
        for job in list(active):
            code = job['process'].poll()
            if code is None: continue
            job['handle'].close(); active.remove(job)
            records[job['key']].update(status='complete' if code==0 else 'failed',exit_code=code)
            if code and 'out of memory' in Path(records[job['key']]['log']).read_text().lower():
                capacity = [f'cuda:{i}' for i in devices]
            if code and attempts[job['key']] < 2:
                pending.append((job['mode'],job['seed']))
        write_json(logroot/f'{stage}_status.json', records)
        if active: time.sleep(10)
    if any(v['status']!='complete' for v in records.values()):
        raise RuntimeError(f'{stage} has failed jobs; see independent logs')


def test(root, device):
    lock = json.loads((root/'selection/spatiotemporal_lock.json').read_text())
    if lock['selected']=='base': return
    for path, digest in lock['checkpoints'].items():
        if sha256(Path(path)) != digest: raise RuntimeError('Locked checkpoint changed')
    for mode in lock['test_models']:
        for seed in SEEDS:
            if mode=='base': reuse_base(seed,'test',root)
            else: evaluate_deployed(mode,seed,'test',device,root)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=['verify','smoke','train','refit','select','test','report','pipeline'],required=True)
    p.add_argument('--root',type=Path,default=ROOT)
    p.add_argument('--mode',choices=['constant','conditioned'])
    p.add_argument('--seed',type=int,choices=SEEDS)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--devices',type=int,nargs='+',default=[0,1,2,3])
    p.add_argument('--slots-per-gpu',type=int,choices=[1,2],default=1)
    args = p.parse_args(); root=args.root
    root.mkdir(parents=True,exist_ok=True)
    if args.stage == 'verify': verify(root)
    elif args.stage == 'smoke': smoke(root,args.device)
    elif args.stage in ('train','refit'):
        if args.mode is not None and args.seed is not None:
            (train if args.stage=='train' else refit)(args.mode,args.seed,args.device,root)
        else:
            matrix(root,args.stage,args.devices,args.slots_per_gpu)
    elif args.stage=='select': selection(root)
    elif args.stage=='test': test(root,args.device)
    elif args.stage=='report': report(root)
    else:
        write_json(root/'background/pipeline_status.json', {'pid':os.getpid(),'stage':'verify','status':'running'})
        verify(root); smoke(root,args.device)
        peaks = [json.loads((run_dir(m,SEEDS[0],root/'smoke')/'result.json').read_text())['peak_memory_bytes']
                 for m in ('constant','conditioned')]
        slots = min(args.slots_per_gpu, 1 if max(peaks)>10*1024**3 else 2)
        for stage in ('train','refit'):
            write_json(root/'background/pipeline_status.json', {'pid':os.getpid(),'stage':stage,'status':'running','slots_per_gpu':slots})
            matrix(root,stage,args.devices,slots)
        write_json(root/'background/pipeline_status.json', {'pid':os.getpid(),'stage':'select','status':'running'})
        selection(root)
        test(root,args.device)
        from football_hgt_targets_v4.spatiotemporal_verify import diagnostics_and_efficiency
        diagnostics_and_efficiency(root,args.device)
        report(root)
        write_json(root/'background/pipeline_status.json', {'pid':os.getpid(),'stage':'complete','status':'complete'})


if __name__=='__main__':
    try:
        main()
    except BaseException as exc:
        if '--stage' in sys.argv and sys.argv[sys.argv.index('--stage')+1]=='pipeline':
            root = Path(sys.argv[sys.argv.index('--root')+1]) if '--root' in sys.argv else ROOT
            write_json(root/'background/pipeline_status.json',{'pid':os.getpid(),'status':'failed','error':repr(exc)})
        raise
