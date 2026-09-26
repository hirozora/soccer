import copy
import math
from types import SimpleNamespace
import pytest
import torch
from torch.nn import functional as F

from test_partial_l2 import artifacts, batch
from football_hgt_targets_v4.model import build_partial_l2_model
from football_hgt_targets_v4.subevent_auxiliary import (
    SubeventAuxiliaryHGT as AuxiliaryHGT, AuxiliaryCollate, SUBTYPES,
    label, auxiliary_loss, grouped_logits, inference_model, AUX_WEIGHT)
from football_hgt_targets_v4.fixed_budget_loss import fixed_budget_loss
from football_hgt_targets_v4.fixed_budget_study import ALL_TASKS

torch.set_num_threads(1)
torch.backends.mkldnn.enabled = False

@pytest.mark.parametrize('mode', ['coarse', 'fine'])
def test_initial_equivalence_and_rng(artifacts, batch, mode):
    torch.manual_seed(20260715)
    base = build_partial_l2_model(artifacts).eval()
    rng = torch.get_rng_state().clone()
    torch.manual_seed(20260715)
    model = AuxiliaryHGT(artifacts, mode).eval()
    assert torch.equal(rng, torch.get_rng_state())
    for key, value in base.state_dict().items():
        assert torch.equal(value, model.state_dict()[key]), key
    with torch.no_grad():
        left, right = base(batch), model(batch)
        assert max((left[k] - right[k]).abs().max().item() for k in left) < 1e-6
        l = fixed_budget_loss(left, batch, artifacts, ALL_TASKS)[0]
        r = fixed_budget_loss(right, batch, artifacts, ALL_TASKS)[0]
        assert abs(float(l-r)) < 1e-6
    assert sum(p.numel() for p in model.parameters()) - sum(p.numel() for p in base.parameters()) == 910


def test_train_dropout_order(artifacts, batch):
    torch.manual_seed(55); base = build_partial_l2_model(artifacts).train()
    torch.manual_seed(55); model = AuxiliaryHGT(artifacts, 'fine').train()
    state = torch.get_rng_state()
    with torch.no_grad(): left = base(batch)
    torch.set_rng_state(state)
    with torch.no_grad(): right = model(batch)
    assert max(float((left[k]-right[k]).abs().max()) for k in left) < 1e-6


def test_target_information_not_read(artifacts, batch):
    model = AuxiliaryHGT(artifacts,'fine',dropout=0).eval()
    modified = dict(batch)
    modified['targets'] = {k:torch.zeros_like(v) for k,v in batch['targets'].items()}
    with torch.no_grad():
        a,b = model(batch),model(modified)
    assert all(torch.equal(a[k],b[k]) for k in a)


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
    model=AuxiliaryHGT(artifacts,'coarse',dropout=0).eval()
    with torch.no_grad():
        predictions=model(batch)
        total,_,core=fixed_budget_loss(predictions,batch,artifacts,ALL_TASKS)
        frame,_=_merge_prediction_frames(predictions,batch,total)
    assert abs(core_loss(frame)-float(core))<1e-6


def test_optimizer_and_rng_exact_resume(artifacts, batch, tmp_path):
    model=AuxiliaryHGT(artifacts,'fine').train()
    optimizer=torch.optim.AdamW(model.parameters(),lr=9e-4)
    def step(m,o):
        o.zero_grad(set_to_none=True)
        pred=m(batch)
        loss=fixed_budget_loss(pred,batch,artifacts,ALL_TASKS)[0] + AUX_WEIGHT * auxiliary_loss(pred['aux_logits'], synthetic_targets(1), 'fine')
        loss.backward(); o.step()
        return loss.detach()
    step(model,optimizer)
    path=tmp_path/'last.pt'
    torch.save({'model':model.state_dict(),'optimizer':optimizer.state_dict(),'rng':torch.get_rng_state()},path)
    expected=step(model,optimizer)
    restored=AuxiliaryHGT(artifacts,'fine').train()
    opt=torch.optim.AdamW(restored.parameters(),lr=9e-4)
    checkpoint=torch.load(path,weights_only=False)
    restored.load_state_dict(checkpoint['model']);opt.load_state_dict(checkpoint['optimizer'])
    torch.set_rng_state(checkpoint['rng'])
    actual=step(restored,opt)
    assert torch.equal(expected,actual)
    assert all(torch.equal(v,restored.state_dict()[k]) for k,v in model.state_dict().items())

def test_test_gate(tmp_path):
    from football_hgt_targets_v4.subevent_training import require_test, frame_path
    with pytest.raises(RuntimeError): require_test(tmp_path)
    with pytest.raises(RuntimeError): frame_path('fine', 20260715, 'test', tmp_path)


def test_selection_requires_event_both_references_and_guards():
    from football_hgt_targets_v4.subevent_reporting import choose
    from football_hgt_targets_v4.five_task_reporting import METRICS
    def comparison(gain=0.):
        result = {k: {'difference': 0., 'per_seed': [0.]*3, 'ci_low': 0., 'ci_high': 0.} for k in METRICS}
        result['event_macro_f1'] = {'difference': gain, 'per_seed': [gain]*3, 'ci_low': gain*.8, 'ci_high': gain*1.2}
        return result
    pairs = {'coarse-base': comparison(.007), 'fine-base': comparison(.009),
             'fine-coarse': comparison(.002)}
    losses = {'base': .03, 'coarse': .02, 'fine': .01}
    assert choose(pairs, losses, list(losses))[0] == 'coarse'
    pairs['fine-coarse'] = comparison(.006)
    assert choose(pairs, losses, list(losses))[0] == 'fine'
    pairs['fine-coarse']['event_accuracy']['difference'] = -.011
    assert choose(pairs, losses, list(losses))[0] == 'coarse'
    pairs['coarse-base']['player_top1']['difference'] = -.006
    assert choose(pairs, losses, list(losses))[0] == 'base'



def synthetic_targets(n=14):
    fine = torch.arange(n) % 14
    return {'aux_subtype': fine, 'aux_coarse': torch.where(fine < 4, 0, torch.where(fine < 7, 1, 2)),
            'aux_mask': torch.ones(n, dtype=torch.bool)}


def test_label_mapping_and_reject_unknown():
    for index, (event, subtype) in enumerate(SUBTYPES):
        assert label(event, subtype) == (index, 0 if index < 4 else 1 if index < 7 else 2)
    assert label(2, 20) == (-1, -1)
    for event, subtype in [(1, None), (7, 80), (8, 999)]:
        with pytest.raises(ValueError): label(event, subtype)


def test_coarse_fine_exact_decomposition_and_empty():
    torch.manual_seed(42)
    logits = torch.randn(14, 14, requires_grad=True)
    targets = synthetic_targets()
    coarse = auxiliary_loss(logits, targets, 'coarse')
    fine = auxiliary_loss(logits, targets, 'fine')
    within = []
    for i, (lo, hi) in enumerate([(0,4)]*4 + [(4,7)]*3 + [(7,14)]*7):
        within.append(F.cross_entropy(logits[i:i+1, lo:hi], torch.tensor([i-lo])))
    assert torch.allclose(fine, coarse + torch.stack(within).mean()/math.log(14), atol=1e-7)
    targets['aux_mask'].zero_()
    empty = auxiliary_loss(logits, targets, 'fine')
    empty.backward()
    assert empty.item() == 0 and torch.equal(logits.grad, torch.zeros_like(logits))
    assert AUX_WEIGHT == .02


def test_labels_only_use_successor_and_preserve_unknown_player(artifacts):
    def base(samples):
        return {'targets': {'player_mask': torch.tensor([False]*len(samples))}}
    event = {'event_type_index': torch.tensor([7, 0, 6]),
             'subevent_type_index': torch.tensor([artifacts.subevent_type_ids.index(x) for x in [85,10,72]])}
    sample = SimpleNamespace(graph={'node_stores': {'event':event}}, current_event_index=0,
        target_event_index=1, target=SimpleNamespace(raw_event_10=0))
    wrapper = AuxiliaryCollate(base, artifacts)
    actual = wrapper([sample])
    assert actual['targets']['aux_subtype'].item() == 0
    assert actual['targets']['aux_mask'].item()
    assert not actual['targets']['player_mask'].item()
    event['subevent_type_index'][2] = -999
    assert wrapper([sample])['targets']['aux_subtype'].item() == 0


def test_aux_gradient_paths_single_forward_and_stripped_inference(artifacts, batch):
    model = AuxiliaryHGT(artifacts, 'fine', dropout=0).eval()
    layers = [model.convolutions[0], model.convolutions[1], model.player_convolution]
    counts = [0,0,0]
    def hook(i):
        def called(*args): counts[i] += 1
        return called
    handles = [layer.register_forward_hook(hook(i)) for i,layer in enumerate(layers)]
    pred = model(batch)
    assert counts == [1,1,1]
    for h in handles: h.remove()
    auxiliary_loss(pred['aux_logits'], synthetic_targets(1), 'fine').backward()
    for module in [model.auxiliary_head, layers[0], layers[1], model.context_projection]:
        assert any(p.grad is not None and p.grad.abs().sum()>0 for p in module.parameters())
    assert all(p.grad is None for p in layers[2].parameters())
    model.zero_grad(set_to_none=True)
    fixed_budget_loss(model(batch), batch, artifacts, ALL_TASKS)[1]['player'].backward()
    assert all(p.grad is None for p in layers[1].parameters())
    assert all(p.grad is None for p in model.auxiliary_head.parameters())
    stripped = inference_model(artifacts, model.state_dict()).eval()
    with torch.no_grad():
        actual = stripped(batch)
    assert set(actual) == set(pred) - {'aux_logits'}
    assert max(float((actual[k]-pred[k]).abs().max()) for k in actual) < 1e-6


def test_coarse_fine_initialization_identical(artifacts):
    torch.manual_seed(92); coarse = AuxiliaryHGT(artifacts, 'coarse')
    torch.manual_seed(92); fine = AuxiliaryHGT(artifacts, 'fine')
    assert all(torch.equal(v,fine.state_dict()[k]) for k,v in coarse.state_dict().items())


def test_original_validation_losses_ignore_auxiliary(artifacts, batch):
    model = AuxiliaryHGT(artifacts, 'fine').eval()
    with torch.no_grad():
        pred = model(batch)
        before = fixed_budget_loss(pred, batch, artifacts, ALL_TASKS)
        pred['aux_logits'].fill_(1e5)
        after = fixed_budget_loss(pred, batch, artifacts, ALL_TASKS)
    assert torch.equal(before[0],after[0]) and torch.equal(before[2],after[2])

