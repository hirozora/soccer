"""Target-only raw subevent supervision; the five inference heads are unchanged."""
from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F

from .model import PartialL2FiveTaskHGT, build_partial_l2_model

MODES = ('coarse', 'fine')
SUBTYPES = tuple([(1, s) for s in range(10, 14)] +
                 [(7, s) for s in range(70, 73)] + [(8, s) for s in range(80, 87)])
GROUPS = ((0, 4), (4, 7), (7, 14))
AUX_WEIGHT = .02


def label(event_id, subevent_id):
    if event_id not in (1, 7, 8):
        return -1, -1
    try:
        index = SUBTYPES.index((event_id, subevent_id))
    except ValueError as error:
        raise ValueError(f'Undefined supervised subtype: {(event_id, subevent_id)}') from error
    return index, (1, 7, 8).index(event_id)


@dataclass
class AuxiliaryCollate:
    base: object
    artifacts: object

    def __call__(self, samples):
        batch = self.base(samples)
        labels = []
        for sample in samples:
            event = sample.graph['node_stores']['event']
            i = sample.target_event_index
            if i <= sample.current_event_index:
                raise ValueError('Auxiliary supervision must target the next event')
            raw = int(event['event_type_index'][i])
            if raw != sample.target.raw_event_10:
                raise ValueError('Raw-10 target alignment mismatch')
            labels.append(label(self.artifacts.event_type_ids[raw],
                self.artifacts.subevent_type_ids[int(event['subevent_type_index'][i])]))
        batch['targets']['aux_subtype'] = torch.tensor([v[0] for v in labels], dtype=torch.long)
        batch['targets']['aux_coarse'] = torch.tensor([v[1] for v in labels], dtype=torch.long)
        batch['targets']['aux_mask'] = batch['targets']['aux_subtype'] >= 0
        return batch


def grouped_logits(logits):
    return torch.stack([logits[:, lo:hi].logsumexp(-1) for lo, hi in GROUPS], -1)


def auxiliary_loss(logits, targets, mode):
    if mode not in MODES:
        raise ValueError(mode)
    mask = targets['aux_mask']
    if not bool(mask.any()):
        return logits.sum() * 0.
    selected = logits[mask]
    if mode == 'fine':
        loss = F.cross_entropy(selected, targets['aux_subtype'][mask])
    else:
        loss = F.cross_entropy(grouped_logits(selected), targets['aux_coarse'][mask])
    return loss / math.log(14)


class SubeventAuxiliaryHGT(PartialL2FiveTaskHGT):
    def __init__(self, artifacts, mode, dropout=.1):
        if mode not in MODES:
            raise ValueError(mode)
        super().__init__(artifacts, 'five_f80', dropout=dropout)
        self.aux_mode = mode
        # CPU construction under a fork preserves the public model/dropout RNG stream.
        with torch.random.fork_rng(devices=[]):
            self.auxiliary_head = nn.Linear(64, 14)

    def forward_with_contexts(self, batch):
        predictions, contexts = super().forward_with_contexts(batch)
        predictions['aux_logits'] = self.auxiliary_head(contexts['f80'])
        return predictions, contexts


def public_state(model):
    state = model if isinstance(model, dict) else model.state_dict()
    return {k: v for k, v in state.items() if not k.startswith('auxiliary_head.')}


def inference_model(artifacts, state):
    with torch.random.fork_rng(devices=[]):
        model = build_partial_l2_model(artifacts)
    model.load_state_dict(public_state(state))
    return model
