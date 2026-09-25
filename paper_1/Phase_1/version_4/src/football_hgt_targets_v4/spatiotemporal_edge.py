"""Continuous pair-conditioned gates on existing causal Event adjacency edges."""

import math
import torch
from torch import nn

from .model import PartialL2FiveTaskHGT, POSSESSION_HGT_METADATA
from .state_gating import StateAwareHGTConv

MODES = ("constant", "conditioned")
NEXT = ("event", "next", "event")
TEAM = ("event", "performed_by_team", "team")


def pair_features(start, end, start_valid, end_valid, teams, seconds, pairs):
    """Five values per pair, in the destination actor team's coordinate frame."""
    src, dst = pairs
    valid = (start_valid | end_valid).bool()
    positions = torch.where(end_valid.bool()[:, None], end, start)
    geometry_valid = valid[src] & valid[dst] & (teams[src] >= 0) & (teams[dst] >= 0)
    source = torch.where((teams[src] == teams[dst])[:, None], positions[src], 1. - positions[src])
    delta = positions[dst] - source
    delta = torch.where(geometry_valid[:, None], delta, torch.zeros_like(delta))
    meters = delta * delta.new_tensor([105., 68.])
    dt = seconds[dst] - seconds[src]
    if not torch.isfinite(dt).all():
        raise ValueError("Non-finite event time")
    x = torch.stack((torch.log1p(dt.clamp(0., 7200.)) / math.log(7201.), delta[:, 0], delta[:, 1],
                     meters.norm(dim=-1) / math.hypot(105., 68.), geometry_valid.float()), -1)
    if not torch.isfinite(x).all():
        raise ValueError("Non-finite edge features")
    return x


def edge_bundle(graph):
    event = graph["event"]
    edges = graph.edge_index_dict
    pairs = edges[NEXT]
    n = event.num_nodes
    keys = pairs[0] * n + pairs[1]
    unique, _ = torch.unique(keys, sorted=True, return_inverse=True)
    pairs = torch.stack((unique // n, unique % n))
    src, dst = pairs
    if bool((event.batch[src] != event.batch[dst]).any()) or bool((event.source_index[dst] - event.source_index[src] != 1).any()):
        raise ValueError("Event adjacency is not causal/consecutive")
    same_period = event.period_index[src] == event.period_index[dst]
    teams = torch.full((n,), -1, dtype=torch.long, device=src.device)
    te = edges[TEAM]
    known = graph["team"].vocab_index[te[1]] > 0
    teams[te[0, known]] = te[1, known]
    features = pair_features(event.start_position, event.end_position, event.start_position_mask,
                             event.end_position_mask, teams, event.absolute_seconds, pairs)
    features = torch.where(same_period[:, None], features, torch.zeros_like(features))
    mappings = {}
    for relation, index in edges.items():
        mapping = torch.full((index.shape[1],), -1, dtype=torch.long, device=index.device)
        if relation == NEXT or (relation[0] == relation[2] == "event" and relation[1].startswith("gap_")):
            edge_keys = index[0] * n + index[1]
            where = torch.searchsorted(unique, edge_keys)
            if edge_keys.numel() and (not unique.numel() or bool((where >= len(unique)).any())
                                     or not torch.equal(unique[where], edge_keys)):
                raise ValueError("Time bucket edge has no matching next edge")
            mapping = torch.where(same_period[where], where, mapping)
        mappings[relation] = mapping
    dt = event.absolute_seconds[dst] - event.absolute_seconds[src]
    return {"features": features, "mappings": mappings, "pairs": pairs, "eligible": same_period,
            "negative_time": (dt < 0) & same_period, "clipped_time": (dt > 7200) & same_period}


class EdgeController(nn.Module):
    def __init__(self, mode):
        super().__init__()
        self.mode = mode
        self.network = nn.Sequential(nn.Linear(5, 16), nn.GELU(), nn.Linear(16, 1))
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, bundle):
        features = bundle["features"]
        if self.mode == "constant":
            features = torch.zeros_like(features)
        return 2. * self.network(features).squeeze(-1).sigmoid()


class EdgeHGTConv(StateAwareHGTConv):
    def __init__(self, source, mode):
        super().__init__(source.in_channels, source.out_channels, POSSESSION_HGT_METADATA, source.heads)
        self.gate_controller = EdgeController(mode)
        missing, unexpected = self.load_state_dict(source.state_dict(), strict=False)
        if unexpected or any(not k.startswith("gate_controller.") for k in missing):
            raise RuntimeError("HGT source mismatch")

    def forward(self, x_dict, edge_index_dict, bundle, node_batch_dict):
        self._mappings = bundle["mappings"]
        return super().forward(x_dict, edge_index_dict, bundle, node_batch_dict)

    def edge_gate_vector(self, edge_index_dict, node_batch_dict, gate_matrix, edge_type_order=None):
        # Index zero is reserved for unaffected edges, including period breaks.
        padded = torch.cat((gate_matrix.new_ones(1), gate_matrix))
        return torch.cat([padded[self._mappings[r] + 1] for r in edge_type_order or tuple(edge_index_dict)])


class SpatiotemporalHGT(PartialL2FiveTaskHGT):
    def __init__(self, artifacts, mode, dropout=.1):
        if mode not in MODES:
            raise ValueError(mode)
        super().__init__(artifacts, "five_f80", dropout)
        self.edge_mode = mode
        # Construction cannot advance the baseline's dropout/initialization RNG.
        with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
            self.convolutions[0] = EdgeHGTConv(self.convolutions[0], mode)
            self.convolutions[1] = EdgeHGTConv(self.convolutions[1], mode)
            self.player_convolution = EdgeHGTConv(self.player_convolution, mode)
            self.player_convolution.gate_controller.load_state_dict(self.convolutions[1].gate_controller.state_dict())
        self.last_bundle = None

    def encode_partial_views(self, batch):
        if set(batch["graphs"]) != {"f80"}:
            raise ValueError("Spatiotemporal experiment is F80-only")
        graph = batch["graphs"]["f80"]
        bundle = edge_bundle(graph)
        self.last_bundle = bundle
        edges = dict(graph.edge_index_dict)
        batches = {t: graph[t].batch for t in graph.node_types}

        def layer(states, conv, norms):
            updated = conv(states, edges, bundle, batches)
            return {t: norms[t](s + self.dropout(updated[t])) if t in updated else s for t, s in states.items()}

        shared = layer(self._encode_nodes(graph), self.convolutions[0], self.norms[0])
        main = layer(shared, self.convolutions[1], self.norms[1])
        # Preserve the baseline's context-dropout order before the private layer.
        main_context = self._pool_context(main, graph, self.context_projection)
        private = layer(shared, self.player_convolution, self.player_norms)
        return ({"f80": main_context}, {"f80": main},
                self._pool_context(private, graph, self.player_context_projection), private)

    def gate_diagnostics(self):
        bundle = self.last_bundle
        result = {"pairs": bundle["pairs"].shape[1], "same_period_pairs": int(bundle["eligible"].sum()),
                  "negative_time_count": int(bundle["negative_time"].sum()), "clipped_time_count": int(bundle["clipped_time"].sum())}
        for name, layer in (("shared_l1", self.convolutions[0]), ("main_l2", self.convolutions[1]), ("player_l2", self.player_convolution)):
            g = layer.last_gate_matrix[bundle["eligible"]].detach()
            result[name] = {"mean": float(g.mean()) if g.numel() else None,
                            "min": float(g.min()) if g.numel() else None, "max": float(g.max()) if g.numel() else None}
        return result
