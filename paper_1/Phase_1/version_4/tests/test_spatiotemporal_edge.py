import copy
import pytest
import torch

from test_partial_l2 import artifacts, batch
from football_hgt_targets_v4.model import build_partial_l2_model
from football_hgt_targets_v4.spatiotemporal_edge import SpatiotemporalHGT, EdgeController, edge_bundle, pair_features, NEXT
from football_hgt_targets_v4.fixed_budget_loss import fixed_budget_loss
from football_hgt_targets_v4.fixed_budget_study import ALL_TASKS

torch.set_num_threads(1)
torch.backends.mkldnn.enabled = False


def test_geometry():
    start = torch.tensor([[.2, .3], [.8, .7], [.9, .8]])
    valid = torch.ones(3, dtype=torch.bool)
    pairs = torch.tensor([[0, 1], [1, 2]])
    x = pair_features(start, start, valid, valid, torch.tensor([0, 1, 1]), torch.tensor([0., 2., 8000.]), pairs)
    assert torch.allclose(x[0, 1:4], torch.zeros(3), atol=1e-7)
    assert torch.allclose(x[1, 1:3], torch.tensor([.1, .1]))
    assert x[1, 0] == 1
    end = start.clone(); end[0] = .5
    x = pair_features(start, end, valid, torch.zeros_like(valid), torch.tensor([0, 1, -1]), torch.tensor([0., -1., 2.]), pairs)
    assert x[0, 0] == 0
    assert torch.equal(x[1, 1:], torch.zeros(4))


@pytest.mark.parametrize('mode', ['constant', 'conditioned'])
def test_initial_equivalence_and_rng(artifacts, batch, mode):
    torch.manual_seed(20260715)
    base = build_partial_l2_model(artifacts).eval()
    rng = torch.get_rng_state().clone()
    torch.manual_seed(20260715)
    model = SpatiotemporalHGT(artifacts, mode).eval()
    assert torch.equal(rng, torch.get_rng_state())
    for key, value in base.state_dict().items():
        assert torch.equal(value, model.state_dict()[key]), key
    with torch.no_grad():
        left, right = base(batch), model(batch)
        assert max((left[k] - right[k]).abs().max().item() for k in left) < 1e-6
        l = fixed_budget_loss(left, batch, artifacts, ALL_TASKS)[0]
        r = fixed_budget_loss(right, batch, artifacts, ALL_TASKS)[0]
        assert abs(float(l-r)) < 1e-6
    a, b = model.convolutions[1].gate_controller, model.player_convolution.gate_controller
    for p, q in zip(a.parameters(), b.parameters()):
        assert torch.equal(p, q) and p.data_ptr() != q.data_ptr()
    assert sum(p.numel() for p in model.parameters()) - sum(p.numel() for p in base.parameters()) == 339


def test_edge_scope(batch):
    graph = batch['graphs']['f80'].clone()
    bundle = edge_bundle(graph)
    for relation, mapping in bundle['mappings'].items():
        if relation == NEXT or relation[1].startswith('gap_'):
            for i, pair in enumerate(graph[relation].edge_index.T):
                m = int(mapping[i])
                if m >= 0:
                    assert torch.equal(bundle['pairs'][:, m], pair)
        else:
            assert bool((mapping == -1).all())
    if graph[NEXT].edge_index.shape[1]:
        u, v = graph[NEXT].edge_index[:, -1]
        graph['event'].period_index[v] = graph['event'].period_index[u] + 1
        bundle = edge_bundle(graph)
        assert bundle['mappings'][NEXT][-1] == -1


def test_controller_gradient_sequence():
    controller = EdgeController('conditioned')
    optimizer = torch.optim.SGD(controller.parameters(), lr=.1)
    bundle = {'features': torch.rand(11, 5)}
    controller(bundle).sum().backward()
    assert controller.network[-1].weight.grad.norm() > 0
    assert controller.network[0].weight.grad.norm() == 0
    optimizer.step(); optimizer.zero_grad()
    controller(bundle).sum().backward()
    assert controller.network[0].weight.grad.norm() > 0
    constant = EdgeController('constant')
    assert torch.equal(constant(bundle), constant({'features': torch.randn(11, 5)}))


def test_branches_and_restore(artifacts, batch):
    model = SpatiotemporalHGT(artifacts, 'conditioned', dropout=0).eval()
    counts = [0, 0, 0]
    layers = [model.convolutions[0], model.convolutions[1], model.player_convolution]
    def hook(i):
        def record(*args): counts[i] += 1
        return record
    handles = [layer.register_forward_hook(hook(i)) for i, layer in enumerate(layers)]
    pred = model(batch)
    assert counts == [1, 1, 1]
    for h in handles: h.remove()
    components = fixed_budget_loss(pred, batch, artifacts, ALL_TASKS)[1]
    components['event'].backward()
    assert all(p.grad is None for p in layers[2].parameters())
    model.zero_grad(set_to_none=True)
    components = fixed_budget_loss(model(batch), batch, artifacts, ALL_TASKS)[1]
    components['player'].backward()
    assert all(p.grad is None for p in layers[1].parameters())
    clone = SpatiotemporalHGT(artifacts, 'conditioned', dropout=0).eval()
    clone.load_state_dict(copy.deepcopy(model.state_dict()))
    with torch.no_grad():
        restored = clone(batch)
        assert all(torch.equal(pred[k], restored[k]) for k in pred)


def test_train_dropout_order(artifacts, batch):
    torch.manual_seed(55); base = build_partial_l2_model(artifacts).train()
    torch.manual_seed(55); model = SpatiotemporalHGT(artifacts, 'conditioned').train()
    state = torch.get_rng_state()
    with torch.no_grad(): left = base(batch)
    torch.set_rng_state(state)
    with torch.no_grad(): right = model(batch)
    assert max(float((left[k]-right[k]).abs().max()) for k in left) < 1e-6


def test_target_information_not_read(artifacts, batch):
    model = SpatiotemporalHGT(artifacts,'conditioned',dropout=0).eval()
    modified = dict(batch)
    modified['targets'] = {k:torch.zeros_like(v) for k,v in batch['targets'].items()}
    with torch.no_grad():
        a,b = model(batch),model(modified)
    assert all(torch.equal(a[k],b[k]) for k in a)


def test_selection_same_task_and_guard_contract():
    from football_hgt_targets_v4.spatiotemporal_reporting import choose, METRICS
    def cmp():
        return {k:{'difference':0.,'per_seed':[0.,0.,0.],'ci_low':0.,'ci_high':0.} for k in METRICS}
    def improve(d,k,value):
        d[k]={'difference':value,'per_seed':[value]*3,'ci_low':value*.8,'ci_high':value*1.2}
        if value<0: d[k]['ci_low'],d[k]['ci_high']=value*1.2,value*.8
    a,b,c=cmp(),cmp(),cmp()
    improve(a,'event_macro_f1',.01)
    improve(b,'event_macro_f1',.01)
    improve(c,'position_distance_mae_m',-.4)
    comparisons={'constant-base':a,'conditioned-base':b,'conditioned-constant':c}
    losses={'base':.3,'constant':.2,'conditioned':.1}
    result,eligible,_=choose(comparisons,losses,['base','constant','conditioned'])
    assert result=='constant' and 'conditioned' not in eligible
    improve(c,'event_macro_f1',.006)
    assert choose(comparisons,losses,['base','constant','conditioned'])[0]=='conditioned'
    improve(b,'player_top1',-.006)
    assert choose(comparisons,losses,['base','constant','conditioned'])[0]=='constant'
    assert choose({}, {'base':1.}, ['base'])[0]=='base'


def test_test_access_and_fixed_budget(tmp_path):
    from football_hgt_targets_v4.spatiotemporal_training import require_test, config_for, write_json
    from football_hgt_targets_v4.spatiotemporal_reporting import frame_path
    with pytest.raises(RuntimeError): require_test(tmp_path)
    with pytest.raises(RuntimeError): frame_path('base',20260715,'test',tmp_path)
    write_json(tmp_path/'selection/spatiotemporal_lock.json',{'selected':'base','test_models':[],'checkpoints':{}})
    with pytest.raises(RuntimeError): require_test(tmp_path)
    assert config_for('conditioned',20260715,'cpu',tmp_path).training_budget==24


def test_match_only_bootstrap():
    import pandas as pd
    from football_hgt_targets_v4.spatiotemporal_reporting import compare
    frame = pd.DataFrame({'sample_id':['a','b','c','d'],'match_id':[1,1,2,2], 'current_event_index':[0,1,0,1],
        'event_true':[0,1,0,1],'event_pred':[1,1,1,1],'time_true':[1.]*4,'time_pred':[2.]*4,'time_mask':[True]*4,
        'position_true_x':[.5]*4,'position_true_y':[.5]*4,'position_pred_x':[.6]*4,'position_pred_y':[.5]*4,
        'position_mask':[True]*4,'player_mask':[True]*4,'player_rank':[2.]*4,'team_true':[0,1,0,1],'team_pred':[0,1,0,1]})
    result = compare([frame]*3,[frame]*3,iterations=100)
    assert all(v['difference']==v['ci_low']==v['ci_high']==0 for v in result.values())
    altered=frame.copy();altered['time_pred']=1.
    result=compare([frame]*3,[altered]*3,iterations=100)
    assert result['time_mae_seconds']['difference']==-1
    assert result['time_mae_seconds']['ci_low']==result['time_mae_seconds']['ci_high']==-1


def test_core_loss_matches_existing_definition(artifacts, batch):
    from football_hgt_targets_v4.five_task_training import _merge_prediction_frames
    from football_hgt_targets_v4.spatiotemporal_reporting import core_loss
    model=SpatiotemporalHGT(artifacts,'constant',dropout=0).eval()
    with torch.no_grad():
        predictions=model(batch)
        total,_,core=fixed_budget_loss(predictions,batch,artifacts,ALL_TASKS)
        frame,_=_merge_prediction_frames(predictions,batch,total)
    assert abs(core_loss(frame)-float(core))<1e-6


def test_optimizer_and_rng_exact_resume(artifacts, batch, tmp_path):
    model=SpatiotemporalHGT(artifacts,'conditioned').train()
    optimizer=torch.optim.AdamW(model.parameters(),lr=9e-4)
    def step(m,o):
        o.zero_grad(set_to_none=True)
        loss=fixed_budget_loss(m(batch),batch,artifacts,ALL_TASKS)[0]
        loss.backward(); o.step()
        return loss.detach()
    step(model,optimizer)
    path=tmp_path/'last.pt'
    torch.save({'model':model.state_dict(),'optimizer':optimizer.state_dict(),'rng':torch.get_rng_state()},path)
    expected=step(model,optimizer)
    restored=SpatiotemporalHGT(artifacts,'conditioned').train()
    opt=torch.optim.AdamW(restored.parameters(),lr=9e-4)
    checkpoint=torch.load(path,weights_only=False)
    restored.load_state_dict(checkpoint['model']);opt.load_state_dict(checkpoint['optimizer'])
    torch.set_rng_state(checkpoint['rng'])
    actual=step(restored,opt)
    assert torch.equal(expected,actual)
    assert all(torch.equal(v,restored.state_dict()[k]) for k,v in model.state_dict().items())
