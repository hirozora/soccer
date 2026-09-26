"""Main-L2 pre-softmax edge bias; the standard HGT forward stays unchanged."""
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import HGTConv
from torch_geometric.utils import softmax

from .model import PartialL2FiveTaskHGT, POSSESSION_HGT_METADATA
from .spatiotemporal_edge import edge_bundle, TEAM

MODES = ("constant", "transition")
MEMBERSHIP = ("event", "belongs_to", "possession")


def endpoint_statistics(possession, seconds):
    """Within-view prefix statistics, never a final or anchor-time aggregate."""
    n = len(possession)
    order = torch.argsort(possession, stable=True)
    ids = possession[order]
    rank = torch.arange(n, device=ids.device)
    starts = torch.ones(n, dtype=torch.bool, device=ids.device)
    starts[1:] = ids[1:] != ids[:-1]
    first = torch.where(starts, rank, 0).cummax(0).values
    count = torch.zeros_like(seconds)
    duration = torch.zeros_like(seconds)
    count[order] = (rank - first + 1).to(seconds.dtype)
    duration[order] = (seconds[order] - seconds[order[first]]).clamp(0, 7200)
    valid = possession >= 0
    return torch.where(valid, duration, 0), torch.where(valid, count, 0)


def transition_bundle(graph):
    bundle = edge_bundle(graph)
    event = graph['event']
    u, v = bundle['pairs']
    n = event.num_nodes
    teams = torch.full((n,), -1, device=u.device, dtype=torch.long)
    te = graph[TEAM].edge_index
    known = graph['team'].vocab_index[te[1]] > 0
    teams[te[0, known]] = te[1, known]
    possession = torch.full_like(teams, -1)
    pe = graph[MEMBERSHIP].edge_index
    if pe.shape[1] != torch.unique(pe[0]).numel():
        raise ValueError('An Event has multiple Possessions')
    possession[pe[0]] = pe[1]
    duration, count = endpoint_statistics(possession, event.absolute_seconds)
    stats = torch.stack((torch.log1p(duration) / math.log(7201),
                         torch.log1p(count) / math.log(81)), -1)
    team_valid = (teams[u] >= 0) & (teams[v] >= 0)
    pu, pv = possession[u] >= 0, possession[v] >= 0
    controls = F.one_hot(event.control_state_after_index, 4)
    roles = F.one_hot(event.event_role_index, 6)
    features = torch.cat((bundle['features'],
        torch.stack(((teams[u] == teams[v]) & team_valid, team_valid), -1),
        torch.stack(((possession[u] == possession[v]) & pu & pv, pu, pv), -1),
        controls[u], controls[v], roles[u], roles[v],
        event.switch_confirmed[u, None], event.switch_confirmed[v, None], stats[u], stats[v]), -1).float()
    if features.shape[1] != 36 or not torch.isfinite(features).all():
        raise ValueError('Invalid 36D transition features')
    bundle['features'] = torch.where(bundle['eligible'][:, None], features, 0)
    return bundle


class EdgeBiasController(nn.Module):
    def __init__(self, mode):
        super().__init__()
        if mode not in MODES:
            raise ValueError(mode)
        self.mode = mode
        self.network = nn.Sequential(nn.Linear(36, 16), nn.GELU(), nn.Linear(16, 4))
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, features):
        return self.network(torch.zeros_like(features) if self.mode == 'constant' else features)


class EdgeBiasHGTConv(HGTConv):
    def __init__(self, source, mode):
        super().__init__(source.in_channels, source.out_channels, POSSESSION_HGT_METADATA, heads=source.heads)
        self.load_state_dict(source.state_dict(), strict=True)
        self.bias_controller = EdgeBiasController(mode)
        self.capture_attention = False

    def forward(self, x_dict, edge_index_dict, bundle):
        pair_bias = self.bias_controller(bundle['features'])
        padded = torch.cat((pair_bias.new_zeros(1, self.heads), pair_bias))
        # PyG constructs source offsets and bipartite edges in this dictionary order.
        mappings = torch.cat([bundle['mappings'][r] for r in edge_index_dict])
        self._edge_bias = padded[mappings + 1]
        self.last_pair_bias = pair_bias.detach()
        self._eligible_edges = mappings >= 0
        result = super().forward(x_dict, edge_index_dict)
        self._edge_bias = None
        return result

    def message(self, k_j, q_i, v_j, edge_attr, index, ptr, size_i):
        score = (q_i * k_j).sum(-1) * edge_attr / math.sqrt(q_i.size(-1))
        if score.shape != self._edge_bias.shape:
            raise ValueError('HGT edge ordering/shape mismatch')
        attention = softmax(score + self._edge_bias, index, ptr, size_i)
        if self.capture_attention:
            self.last_attention = attention.detach()
            self.last_destination = index.detach()
        return (v_j * attention.unsqueeze(-1)).view(-1, self.out_channels)


class TransitionECAHGT(PartialL2FiveTaskHGT):
    def __init__(self, artifacts, mode, dropout=.1):
        super().__init__(artifacts, 'five_f80', dropout)
        self.eca_mode = mode
        with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
            self.convolutions[1] = EdgeBiasHGTConv(self.convolutions[1], mode)

    def encode_partial_views(self, batch):
        if set(batch['graphs']) != {'f80'}:
            raise ValueError('ECA requires one F80 view')
        graph = batch['graphs']['f80']
        bundle = transition_bundle(graph)
        self.last_bundle = bundle
        edges = dict(graph.edge_index_dict)
        shared = self._apply_layer(self._encode_nodes(graph), edges, self.convolutions[0], self.norms[0], self.dropout)
        updated = self.convolutions[1](shared, edges, bundle)
        main = {t: self.norms[1][t](s + self.dropout(updated[t])) if t in updated else s
                for t, s in shared.items()}
        context = self._pool_context(main, graph, self.context_projection)
        private = self._apply_layer(shared, edges, self.player_convolution, self.player_norms, self.dropout)
        return ({'f80': context}, {'f80': main},
                self._pool_context(private, graph, self.player_context_projection), private)


def public_state(model):
    return {k: v for k, v in model.state_dict().items() if '.bias_controller.' not in k}
