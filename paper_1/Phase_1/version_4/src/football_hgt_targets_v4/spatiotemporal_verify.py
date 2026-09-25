"""Full-batch equivalence, branch costs and held-out validation gate diagnostics."""
import json
import time
import numpy as np
import pandas as pd
import torch
from football_benchmark.protocol import ProtocolArtifacts
from .constants import FEASIBILITY_ARTIFACT
from .model import build_partial_l2_model
from .spatiotemporal_edge import SpatiotemporalHGT
from .spatiotemporal_training import ROOT, SEEDS, config_for, load_backbone, run_dir
from .fixed_budget_training import _loader_config, _common_hash
from .five_task_training import _loader
from .training import _move_batch_to_device, set_seed
from .fixed_budget_loss import fixed_budget_loss
from .fixed_budget_study import ALL_TASKS
from .partial_sharing_study import training_dir as base_dir
from .position_head_refit import write_json, ROOT as POSITION_ROOT


def _sync(device):
    if str(device).startswith('cuda'): torch.cuda.synchronize(device)


def _batch(root, device):
    artifacts = ProtocolArtifacts.load(FEASIBILITY_ARTIFACT)
    cfg = config_for('conditioned', SEEDS[0], device, root, smoke=True, workers=0)
    loader = _loader('train', _loader_config(cfg), artifacts, False)
    tick = time.perf_counter(); raw = next(iter(loader)); cpu = time.perf_counter()-tick
    tick = time.perf_counter(); batch = _move_batch_to_device(raw,torch.device(device)); _sync(device)
    return artifacts, raw, batch, cpu, time.perf_counter()-tick


def verify_device(root=ROOT, device='cpu'):
    artifacts, raw, batch, cpu, h2d = _batch(root,device)
    checks = []
    for seed in SEEDS:
        set_seed(seed); base = build_partial_l2_model(artifacts).to(device).eval()
        expected_hash = json.loads((base_dir(seed)/'result.json').read_text())['initial_common_sha256']
        if _common_hash(base) != expected_hash: raise RuntimeError('Old baseline initializer changed')
        with torch.no_grad(): expected = base(batch)
        for mode in ('constant','conditioned'):
            set_seed(seed); model = SpatiotemporalHGT(artifacts,mode).to(device).eval()
            if _common_hash(model)!=expected_hash: raise RuntimeError('Common initializer mismatch')
            with torch.no_grad(): actual = model(batch)
            errors = {k:float((expected[k]-actual[k]).abs().max()) for k in expected}
            left = fixed_budget_loss(expected,batch,artifacts,ALL_TASKS)[0]
            right = fixed_budget_loss(actual,batch,artifacts,ALL_TASKS)[0]
            if max(errors.values()) >= 1e-6 or abs(float(left-right)) >= 1e-6:
                raise RuntimeError(f'Full-batch {device} equivalence failed: {errors}')
            # Keep dropout random streams identical for the first training forward.
            base.train(); model.train(); set_seed(seed+9)
            rng = torch.get_rng_state(); cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
            with torch.no_grad(): a = base(batch)
            torch.set_rng_state(rng)
            if cuda_rng: torch.cuda.set_rng_state_all(cuda_rng)
            with torch.no_grad(): b = model(batch)
            if max(float((a[k]-b[k]).abs().max()) for k in a) >= 1e-6:
                raise RuntimeError('Train/dropout equivalence failed')
            base.eval()
            checks.append({'seed':seed,'mode':mode,'errors':errors,'common_hash':expected_hash})
            del model
        del base
    graph = raw['graphs']['f80']
    write_json(root/'verification/device.json', {'passed':True,'device':device,'checks':checks,
        'samples':len(raw['sample_ids']),'nodes':sum(graph[t].num_nodes for t in graph.node_types),
        'edges':sum(e.shape[1] for e in graph.edge_index_dict.values()),'cpu_collate_seconds':cpu,'h2d_seconds':h2d})


@torch.no_grad()
def diagnostics_and_efficiency(root=ROOT, device='cpu'):
    artifacts, raw, batch, cpu, h2d = _batch(root,device)
    from .event_posterior_online import IntegratedEventModel
    from .oracle_dependency import load_roster_team_map
    roster = load_roster_team_map()
    measurements = []; gates = []
    for mode in ('base','constant','conditioned'):
        for seed in SEEDS:
            if mode=='base':
                model = build_partial_l2_model(artifacts)
                saved = torch.load(base_dir(seed)/'best_guarded_core.pt',map_location='cpu',weights_only=False)
                model.load_state_dict(saved['model']); model=model.to(device).eval()
            else:
                if not (run_dir(mode,seed,root)/'best_guarded_core.pt').exists(): continue
                model,_ = load_backbone(mode,seed,device,root)
            for _ in range(3): model(batch)
            if str(device).startswith('cuda'):
                torch.cuda.reset_peak_memory_stats(device)
            timings=[]
            for _ in range(10):
                _sync(device); start=time.perf_counter(); model(batch); _sync(device)
                timings.append(time.perf_counter()-start)
            latency=float(np.median(timings))
            measurements.append({'mode':mode,'seed':seed,'cpu_collate_seconds':cpu,'h2d_seconds':h2d,
                'forward_seconds':latency,
                'parameters':sum(p.numel() for p in model.parameters()), 'hgt_layer_calls':3,
                'peak_memory_bytes':torch.cuda.max_memory_allocated(device) if str(device).startswith('cuda') else 0,
                'scope':'backbone forward and separately timed Position-Refit/TC composed inference'})
            if mode!='base':
                bundle=model.last_bundle; mask=bundle['eligible']; features=bundle['features'][mask]
                dt=torch.expm1(features[:,0]*np.log(7201.)).cpu().numpy()
                distance=features[:,3].cpu().numpy()*np.hypot(105,68)
                for name, layer in [('shared_l1',model.convolutions[0]),('main_l2',model.convolutions[1]),('player_l2',model.player_convolution)]:
                    g=layer.last_gate_matrix[mask].cpu().numpy()
                    for i in range(len(g)):
                        gates.append({'mode':mode,'seed':seed,'layer':name,'seconds':float(dt[i]),
                            'distance_m':float(distance[i]),'position_valid':bool(features[i,4]),'gate':float(g[i])})
                write_json(root/'diagnostics'/mode/f'seed{seed}.json',model.gate_diagnostics())
            head_path = (POSITION_ROOT/'training'/f'seed{seed}'/'best_position.pt' if mode=='base'
                         else run_dir(mode,seed,root)/'position_refit/best_position.pt')
            model.position_head.load_state_dict(torch.load(head_path,map_location='cpu',weights_only=False)['head'])
            composed = IntegratedEventModel(model,artifacts,roster).to(device).eval()
            composed(batch)
            composed_timings=[]
            for _ in range(5):
                _sync(device); tick=time.perf_counter(); composed(batch); _sync(device)
                composed_timings.append(time.perf_counter()-tick)
            total=cpu+h2d+float(np.median(composed_timings))
            measurements[-1].update(composed_forward_seconds=float(np.median(composed_timings)),
                full_inference_seconds=total,samples_per_second=len(raw['sample_ids'])/total)
            del composed, model
            if str(device).startswith('cuda'): torch.cuda.empty_cache()
    out=root/'efficiency'; out.mkdir(parents=True,exist_ok=True)
    frame=pd.DataFrame(measurements)
    base=frame[frame['mode']=='base'].set_index('seed')
    frame['relative_forward']=[r.forward_seconds/base.loc[r.seed,'forward_seconds'] for r in frame.itertuples()]
    frame['relative_memory']=[r.peak_memory_bytes/max(1,base.loc[r.seed,'peak_memory_bytes']) for r in frame.itertuples()]
    frame['low_cost_pass']=(frame.relative_forward<=1.20)&(frame.relative_memory<=1.10)
    frame.to_csv(out/'measurements.csv',index=False)
    if gates:
        df=pd.DataFrame(gates)
        df['time_group']=pd.cut(df.seconds,[-1,2,5,15,60,7201],right=False)
        df['distance_group']=pd.cut(df.distance_m,[-1,5,10,20,40,60,200],right=False)
        df['time_group']=df['time_group'].astype(str)
        df['distance_group']=df['distance_group'].astype(str)
        df.to_parquet(out/'gate_sample.parquet',index=False)
        for group in ('time_group','distance_group'):
            df.groupby(['mode','seed','layer',group],observed=True).gate.agg(['count','mean','std']).to_csv(out/f'{group}.csv')
    write_json(out/'scope.json',{'sample_split':'train','sample_count':len(raw['sample_ids']),
        'description':'Fixed first full batch; descriptive gate sample, not a population estimate. Warmed forward median of ten.'})
