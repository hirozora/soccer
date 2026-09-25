"""Semantic HGT with interchangeable task-specific output formulations."""

from __future__ import annotations

import copy
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import global_add_pool, global_mean_pool
from torch_geometric.utils import softmax as pyg_softmax

from football_benchmark.constants import ZONE_CENTERS_100
from football_benchmark.models import (
    Projection,
    SEMANTIC_HGT_METADATA,
    SEMANTIC_RELATION_FAMILIES,
    SemanticBenchmarkHGT,
)
from football_benchmark.protocol import ProtocolArtifacts

from .constants import EVENT_METHODS, POSITION_METHODS, TIME_BOUNDARIES, TIME_METHODS
from .possession_data import POSSESSION_EDGE_TYPES
from .five_task_study import FIVE_TASK_MODES
from .state_gating import StateAwareHGTConv
from .age_propagation import (
    PROPAGATION_MODES,
    PROPAGATION_TASKS,
    AgePropagationGate,
    LowRankRelationResidual,
    add_state_residual,
    local_event_ages,
    source_age_mismatch_count,
)


MULTIVIEW_MODES = (
    "f80", "fixed_a", "fixed_b", "sf_a", "sf_b", "recency_sf_b"
)
MULTIVIEW_TASKS = ("event", "time", "position")


def time_bucket_ids(seconds: torch.Tensor) -> torch.Tensor:
    """Map clipped seconds to [0,2), [2,5), [5,15), [15,60]."""

    return torch.bucketize(
        seconds.clamp(0.0, 60.0),
        torch.tensor((2.0, 5.0, 15.0), device=seconds.device, dtype=seconds.dtype),
        right=True,
    )


def decode_bucket_offsets(
    bucket_ids: torch.Tensor, raw_offsets: torch.Tensor
) -> torch.Tensor:
    bounds = torch.tensor(
        TIME_BOUNDARIES, device=raw_offsets.device, dtype=raw_offsets.dtype
    )
    low = bounds[bucket_ids]
    high = bounds[bucket_ids + 1]
    selected = raw_offsets.gather(1, bucket_ids.unsqueeze(1)).squeeze(1)
    return low + torch.sigmoid(selected) * (high - low)


def zone_centers(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.tensor(ZONE_CENTERS_100, device=device, dtype=dtype) / 100.0


def decode_zone_residuals(
    zone_logits: torch.Tensor, raw_residuals: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    zones = zone_logits.argmax(dim=-1)
    rows = torch.arange(zones.shape[0], device=zones.device)
    residual = 0.25 * torch.tanh(raw_residuals[rows, zones])
    position = (
        zone_centers(zones.device, residual.dtype)[zones] + residual
    ).clamp(0.0, 1.0)
    return position, zones


class TargetStudyHGT(SemanticBenchmarkHGT):
    """Exact Semantic HGT backbone with configurable prediction heads."""

    def __init__(
        self,
        artifacts: ProtocolArtifacts,
        task: str,
        method: str,
        joint_methods: dict[str, str] | None = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__(artifacts, contract="unified_lem", dropout=dropout)
        self.task = task
        self.method = method
        self.joint_methods = joint_methods or {}
        if task not in {"event", "time", "position", "joint"}:
            raise ValueError(f"Unknown task {task!r}")
        self._validate_methods()
        if self._uses_time("bucket_offset"):
            self.time_bucket_head = nn.Linear(64, 4)
            self.time_offset_head = nn.Linear(64, 4)
        else:
            self.time_bucket_head = None
            self.time_offset_head = None
        if self._uses_position("zone") or self._uses_position("zone_residual"):
            self.zone_head = nn.Linear(64, 20)
        else:
            self.zone_head = None
        self.zone_residual_head = (
            nn.Linear(64, 40) if self._uses_position("zone_residual") else None
        )

    def _validate_methods(self) -> None:
        methods = self.joint_methods if self.task == "joint" else {self.task: self.method}
        expected = {"event", "time", "position"} if self.task == "joint" else {self.task}
        if set(methods) != expected:
            raise ValueError(f"Methods must define exactly {sorted(expected)}")
        allowed = {
            "event": EVENT_METHODS,
            "time": TIME_METHODS,
            "position": POSITION_METHODS,
        }
        for task, method in methods.items():
            if method not in allowed[task]:
                raise ValueError(f"Unknown {task} method {method!r}")

    def method_for(self, task: str) -> str | None:
        if self.task == "joint":
            return self.joint_methods[task]
        return self.method if self.task == task else None

    def _uses_time(self, method: str) -> bool:
        return self.method_for("time") == method

    def _uses_position(self, method: str) -> bool:
        return self.method_for("position") == method

    def encode_context_and_states(
        self, batch: dict[str, Any]
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        graph = batch["graph"]
        states = self._encode_nodes(graph)
        disabled = set(batch.get("disabled_relation_families", ()))
        unknown = disabled - set(SEMANTIC_RELATION_FAMILIES)
        if unknown:
            raise ValueError(f"Unknown semantic relation families: {sorted(unknown)}")
        disabled_edges = (
            set().union(*(SEMANTIC_RELATION_FAMILIES[name] for name in disabled))
            if disabled
            else set()
        )
        edge_index_dict = {
            edge_type: (
                torch.empty((2, 0), dtype=edge_index.dtype, device=edge_index.device)
                if edge_type in disabled_edges
                else edge_index
            )
            for edge_type, edge_index in graph.edge_index_dict.items()
        }
        for convolution, norms in zip(self.convolutions, self.norms):
            updated = convolution(states, edge_index_dict)
            states = {
                node_type: (
                    norms[node_type](state + self.dropout(updated[node_type]))
                    if node_type in updated
                    else state
                )
                for node_type, state in states.items()
            }
        event_ptr = graph["event"].ptr
        anchors = states["event"][event_ptr[1:] - 1]
        means = global_mean_pool(states["event"], graph["event"].batch)
        context = self.context_projection(torch.cat((anchors, means), dim=-1))
        return context, states

    def encode_context(self, batch: dict[str, Any]) -> torch.Tensor:
        context, _ = self.encode_context_and_states(batch)
        return context

    def forward(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        context = self.encode_context(batch)
        result: dict[str, torch.Tensor] = {}
        if self.method_for("event") is not None:
            result["event_logits"] = self.event_head(context)
        time_method = self.method_for("time")
        if time_method in {"current_huber", "log1p_huber"}:
            log_delta = F.softplus(self.time_head(context).squeeze(-1))
            result["log_delta"] = log_delta
            result["time_seconds"] = torch.expm1(log_delta).clamp(max=60.0)
        elif time_method == "bucket_offset":
            logits = self.time_bucket_head(context)
            offsets = self.time_offset_head(context)
            result["time_bucket_logits"] = logits
            result["time_raw_offsets"] = offsets
            result["time_seconds"] = decode_bucket_offsets(
                logits.argmax(dim=-1), offsets
            )
        position_method = self.method_for("position")
        if position_method == "xy":
            result["position_xy"] = torch.sigmoid(self.position_head(context))
        elif position_method == "zone":
            logits = self.zone_head(context)
            zones = logits.argmax(dim=-1)
            result["zone_logits"] = logits
            result["zone_pred"] = zones
            result["position_xy"] = zone_centers(zones.device, logits.dtype)[zones]
        elif position_method == "zone_residual":
            logits = self.zone_head(context)
            residuals = self.zone_residual_head(context).reshape(-1, 20, 2)
            position, zones = decode_zone_residuals(logits, residuals)
            result["zone_logits"] = logits
            result["zone_raw_residuals"] = residuals
            result["zone_pred"] = zones
            result["position_xy"] = position
        return result


POSSESSION_HGT_METADATA = (
    (*SEMANTIC_HGT_METADATA[0], "possession"),
    (*SEMANTIC_HGT_METADATA[1], *POSSESSION_EDGE_TYPES),
)


def _copy_expanded_hgt(source: nn.Module, target: nn.Module) -> None:
    """Copy all legacy HGT parameters into an expanded metadata instance."""

    source_state = source.state_dict()
    target_state = target.state_dict()
    for name, value in source_state.items():
        if name not in target_state:
            continue
        destination = target_state[name]
        if destination.shape == value.shape:
            destination.copy_(value)
        elif name in {"k_rel.weight", "v_rel.weight"}:
            destination[: value.shape[0]].copy_(value)
    target.load_state_dict(target_state)


class PossessionTargetStudyHGT(TargetStudyHGT):
    """Semantic HGT extended with causally sliced Possession evidence."""

    def __init__(
        self,
        artifacts: ProtocolArtifacts,
        task: str,
        method: str,
        joint_methods: dict[str, str] | None,
        feature_level: str,
        dropout: float = 0.1,
    ) -> None:
        super().__init__(artifacts, task, method, joint_methods, dropout)
        if feature_level not in {"topology", "categorical", "dynamic"}:
            raise ValueError(f"Unknown Possession feature level {feature_level!r}")
        self.possession_feature_level = feature_level
        self.possession_base = nn.Parameter(torch.empty(64))
        nn.init.normal_(self.possession_base, std=0.02)
        self.event_role_embedding = nn.Embedding(6, 4)
        self.actor_relation_embedding = nn.Embedding(4, 4)
        self.control_state_embedding = nn.Embedding(4, 4)
        self.candidate_status_embedding = nn.Embedding(4, 4)
        self.possession_event_projection = Projection(16, 64, dropout)
        self.possession_dynamic_projection = Projection(8, 64, dropout)

        try:
            from torch_geometric.nn import HGTConv
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("torch-geometric is required for HGT") from exc
        expanded = nn.ModuleList()
        for source in self.convolutions:
            target = HGTConv(64, 64, POSSESSION_HGT_METADATA, heads=4)
            _copy_expanded_hgt(source, target)
            expanded.append(target)
        self.convolutions = expanded
        for norms in self.norms:
            norms["possession"] = nn.LayerNorm(64)

    def _encode_nodes(self, graph: Any) -> dict[str, torch.Tensor]:
        states = super()._encode_nodes(graph)
        if self.possession_feature_level in {"categorical", "dynamic"}:
            event = graph["event"]
            categorical = torch.cat(
                (
                    self.event_role_embedding(event.event_role_index),
                    self.actor_relation_embedding(
                        event.actor_relation_to_owner_index
                    ),
                    self.control_state_embedding(event.control_state_after_index),
                    self.candidate_status_embedding(
                        event.candidate_status_after_index
                    ),
                ),
                dim=-1,
            )
            states["event"] = states["event"] + self.possession_event_projection(
                categorical
            )
        possession = graph["possession"]
        possession_state = self.possession_base.unsqueeze(0).expand(
            possession.num_nodes, -1
        )
        if self.possession_feature_level == "dynamic":
            dynamic = torch.cat(
                (
                    (torch.log1p(possession.duration_so_far_seconds) / 5.0).unsqueeze(-1),
                    (possession.event_count_so_far / 80.0).unsqueeze(-1),
                    possession.current_position,
                    possession.current_position_mask.float().unsqueeze(-1),
                    possession.is_closed_as_of_anchor.float().unsqueeze(-1),
                    possession.is_current_active.float().unsqueeze(-1),
                    possession.dynamic_feature_mask.float().unsqueeze(-1),
                ),
                dim=-1,
            )
            possession_state = possession_state + self.possession_dynamic_projection(
                dynamic
            )
        states["possession"] = possession_state
        return states


class TaskViewFusionHGT(PossessionTargetStudyHGT):
    """One shared HGT with hard or static task-specific view fusion."""

    def __init__(
        self,
        artifacts: ProtocolArtifacts,
        mode: str,
        dropout: float = 0.1,
    ) -> None:
        if mode not in MULTIVIEW_MODES:
            raise ValueError(f"Unknown multi-view mode {mode!r}")
        super().__init__(
            artifacts,
            "joint",
            mode,
            {"event": "ce", "time": "current_huber", "position": "xy"},
            "dynamic",
            dropout,
        )
        self.fusion_mode = mode
        self.fusion_logits = nn.ParameterDict()
        if mode == "sf_a":
            tasks = ("time", "position")
        elif mode in {"sf_b", "recency_sf_b"}:
            tasks = MULTIVIEW_TASKS
        else:
            tasks = ()
        for task in tasks:
            probabilities = (0.9, 0.1) if task == "position" else (0.1, 0.9)
            self.fusion_logits[task] = nn.Parameter(
                torch.log(torch.tensor(probabilities, dtype=torch.float32))
            )

    @property
    def required_views(self) -> tuple[str, ...]:
        if self.fusion_mode == "f80":
            return ("f80",)
        if self.fusion_mode == "recency_sf_b":
            return ("lp1", "lp2")
        if self.fusion_mode in {"fixed_b", "sf_b"}:
            return ("p1", "p2")
        return ("f80", "p1", "p2")

    def fusion_weights(self) -> dict[str, torch.Tensor]:
        return {
            task: torch.softmax(logits, dim=-1)
            for task, logits in self.fusion_logits.items()
        }

    def encode_views(
        self, batch: dict[str, Any]
    ) -> dict[str, torch.Tensor]:
        graphs = batch["graphs"]
        missing = set(self.required_views) - set(graphs)
        if missing:
            raise ValueError(f"Missing multi-view graphs: {sorted(missing)}")
        return {
            view: self.encode_context({"graph": graphs[view]})
            for view in self.required_views
        }

    def task_contexts(
        self, contexts: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        if self.fusion_mode == "f80":
            return {task: contexts["f80"] for task in MULTIVIEW_TASKS}
        if self.fusion_mode == "fixed_a":
            return {
                "event": contexts["f80"],
                "time": contexts["p2"],
                "position": contexts["p1"],
            }
        if self.fusion_mode == "fixed_b":
            return {
                "event": contexts["p2"],
                "time": contexts["p2"],
                "position": contexts["p1"],
            }
        weights = self.fusion_weights()
        first, second = (
            ("lp1", "lp2")
            if self.fusion_mode == "recency_sf_b"
            else ("p1", "p2")
        )

        def mix(task: str) -> torch.Tensor:
            return (
                weights[task][0] * contexts[first]
                + weights[task][1] * contexts[second]
            )

        if self.fusion_mode == "sf_a":
            return {
                "event": contexts["f80"],
                "time": mix("time"),
                "position": mix("position"),
            }
        return {task: mix(task) for task in MULTIVIEW_TASKS}

    def context_diagnostics(
        self, contexts: dict[str, torch.Tensor]
    ) -> dict[str, float]:
        pair = (
            ("lp1", "lp2")
            if {"lp1", "lp2"}.issubset(contexts)
            else ("p1", "p2")
        )
        if not set(pair).issubset(contexts):
            return {}
        p1 = contexts[pair[0]]
        p2 = contexts[pair[1]]
        p1_norm = torch.linalg.vector_norm(p1, dim=-1)
        p2_norm = torch.linalg.vector_norm(p2, dim=-1)
        cosine = F.cosine_similarity(p1, p2, dim=-1)
        return {
            "p1_norm_mean": float(p1_norm.detach().mean()),
            "p2_norm_mean": float(p2_norm.detach().mean()),
            "norm_ratio_p1_over_p2": float(
                (p1_norm / p2_norm.clamp_min(1e-12)).detach().mean()
            ),
            "cosine_mean": float(cosine.detach().mean()),
        }

    def forward_with_contexts(
        self, batch: dict[str, Any]
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        contexts = self.encode_views(batch)
        task_context = self.task_contexts(contexts)
        log_delta = F.softplus(self.time_head(task_context["time"]).squeeze(-1))
        predictions = {
            "event_logits": self.event_head(task_context["event"]),
            "log_delta": log_delta,
            "time_seconds": torch.expm1(log_delta).clamp(max=60.0),
            "position_xy": torch.sigmoid(self.position_head(task_context["position"])),
        }
        return predictions, contexts

    def forward(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        predictions, _ = self.forward_with_contexts(batch)
        return predictions


class ActorScaleHGT(PossessionTargetStudyHGT):
    """Single-view next-Team or match-local next-Player predictor."""

    def __init__(
        self,
        artifacts: ProtocolArtifacts,
        actor_task: str,
        dropout: float = 0.1,
    ) -> None:
        if actor_task not in {"team", "player"}:
            raise ValueError(f"Unknown actor task {actor_task!r}")
        super().__init__(
            artifacts,
            "event",
            "ce",
            None,
            "dynamic",
            dropout,
        )
        self.actor_task = actor_task
        # Remove the unused three-target heads from this controlled single-task
        # model so capacity is not silently changed between actor experiments.
        self.event_head = None
        self.time_head = None
        self.position_head = None
        self.team_actor_head = (
            nn.Linear(64, 2) if actor_task == "team" else None
        )
        self.player_actor_scorer = (
            nn.Sequential(
                nn.Linear(128, 64),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(64, 1),
            )
            if actor_task == "player"
            else None
        )

    def forward(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        context, states = self.encode_context_and_states(batch)
        if self.actor_task == "team":
            return {"team_logits": self.team_actor_head(context)}
        player_batch = batch["graph"]["player"].batch
        scores = self.player_actor_scorer(
            torch.cat((context[player_batch], states["player"]), dim=-1)
        ).squeeze(-1)
        return {"player_scores": scores}


class FiveTaskViewHGT(PossessionTargetStudyHGT):
    """One shared HGT for five-task F80, hard-view, or static soft-view training."""

    def __init__(
        self,
        artifacts: ProtocolArtifacts,
        mode: str,
        dropout: float = 0.1,
        player_adapter: bool = False,
    ) -> None:
        if mode not in FIVE_TASK_MODES:
            raise ValueError(f"Unknown five-task mode {mode!r}")
        super().__init__(
            artifacts,
            "joint",
            mode,
            {"event": "ce", "time": "current_huber", "position": "xy"},
            "dynamic",
            dropout,
        )
        self.five_task_mode = mode
        self.player_adapter_enabled = player_adapter
        self.team_actor_head = nn.Linear(64, 2)
        self.player_actor_scorer = nn.Sequential(
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )
        self.player_adapter = None
        if player_adapter:
            self.player_adapter = nn.Sequential(
                nn.Linear(64, 32),
                nn.GELU(),
                nn.Linear(32, 64),
            )
            nn.init.zeros_(self.player_adapter[-1].weight)
            nn.init.zeros_(self.player_adapter[-1].bias)
        self.fusion_logits = nn.ParameterDict()
        if mode == "five_soft":
            for task in MULTIVIEW_TASKS:
                probabilities = (0.9, 0.1) if task == "position" else (0.1, 0.9)
                self.fusion_logits[task] = nn.Parameter(
                    torch.log(torch.tensor(probabilities, dtype=torch.float32))
                )

    @property
    def required_views(self) -> tuple[str, ...]:
        return ("f80",) if self.five_task_mode == "five_f80" else ("f80", "p1", "p2")

    def fusion_weights(self) -> dict[str, torch.Tensor]:
        return {
            task: torch.softmax(logits, dim=-1)
            for task, logits in self.fusion_logits.items()
        }

    def encode_views(
        self, batch: dict[str, Any]
    ) -> tuple[dict[str, torch.Tensor], dict[str, dict[str, torch.Tensor]]]:
        graphs = batch["graphs"]
        missing = set(self.required_views) - set(graphs)
        if missing:
            raise ValueError(f"Missing five-task views: {sorted(missing)}")
        contexts: dict[str, torch.Tensor] = {}
        states: dict[str, dict[str, torch.Tensor]] = {}
        for view in self.required_views:
            contexts[view], states[view] = self.encode_context_and_states(
                {"graph": graphs[view]}
            )
        return contexts, states

    def task_contexts(
        self, contexts: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        if self.five_task_mode == "five_f80":
            return {task: contexts["f80"] for task in (*MULTIVIEW_TASKS, "team", "player")}
        if self.five_task_mode == "five_hard":
            return {
                "event": contexts["p2"],
                "time": contexts["p2"],
                "position": contexts["p1"],
                "team": contexts["f80"],
                "player": contexts["f80"],
            }
        weights = self.fusion_weights()

        def mix(task: str) -> torch.Tensor:
            return weights[task][0] * contexts["p1"] + weights[task][1] * contexts["p2"]

        return {
            "event": mix("event"),
            "time": mix("time"),
            "position": mix("position"),
            "team": contexts["f80"],
            "player": contexts["f80"],
        }

    def context_diagnostics(
        self, contexts: dict[str, torch.Tensor]
    ) -> dict[str, float]:
        if not {"p1", "p2"}.issubset(contexts):
            return {}
        p1, p2 = contexts["p1"], contexts["p2"]
        p1_norm = torch.linalg.vector_norm(p1, dim=-1)
        p2_norm = torch.linalg.vector_norm(p2, dim=-1)
        return {
            "p1_norm_mean": float(p1_norm.detach().mean()),
            "p2_norm_mean": float(p2_norm.detach().mean()),
            "norm_ratio_p1_over_p2": float(
                (p1_norm / p2_norm.clamp_min(1e-12)).detach().mean()
            ),
            "cosine_mean": float(F.cosine_similarity(p1, p2, dim=-1).detach().mean()),
        }

    def forward_with_contexts(
        self, batch: dict[str, Any]
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        contexts, states = self.encode_views(batch)
        task_context = self.task_contexts(contexts)
        log_delta = F.softplus(self.time_head(task_context["time"]).squeeze(-1))
        f80_graph = batch["graphs"]["f80"]
        player_batch = f80_graph["player"].batch
        player_context = task_context["player"]
        if self.player_adapter is not None:
            player_context = player_context + self.player_adapter(player_context)
        player_scores = self.player_actor_scorer(
            torch.cat(
                (player_context[player_batch], states["f80"]["player"]),
                dim=-1,
            )
        ).squeeze(-1)
        predictions = {
            "event_logits": self.event_head(task_context["event"]),
            "log_delta": log_delta,
            "time_seconds": torch.expm1(log_delta).clamp(max=60.0),
            "position_xy": torch.sigmoid(self.position_head(task_context["position"])),
            "team_logits": self.team_actor_head(task_context["team"]),
            "player_scores": player_scores,
        }
        return predictions, contexts

    def forward(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        predictions, _ = self.forward_with_contexts(batch)
        return predictions


class PartialL2FiveTaskHGT(FiveTaskViewHGT):
    """Share layer 1 across views and keep a private F80 layer 2 for Player."""

    def __init__(
        self,
        artifacts: ProtocolArtifacts,
        mode: str = "five_f80",
        dropout: float = 0.1,
    ) -> None:
        super().__init__(artifacts, mode, dropout, player_adapter=False)
        self.player_convolution = copy.deepcopy(self.convolutions[1])
        self.player_norms = copy.deepcopy(self.norms[1])
        self.player_context_projection = copy.deepcopy(self.context_projection)

    @staticmethod
    def _apply_layer(
        states: dict[str, torch.Tensor],
        edge_index_dict: dict[tuple[str, str, str], torch.Tensor],
        convolution: nn.Module,
        norms: nn.ModuleDict,
        dropout: nn.Module,
    ) -> dict[str, torch.Tensor]:
        updated = convolution(states, edge_index_dict)
        return {
            node_type: (
                norms[node_type](state + dropout(updated[node_type]))
                if node_type in updated
                else state
            )
            for node_type, state in states.items()
        }

    @staticmethod
    def _pool_context(
        states: dict[str, torch.Tensor],
        graph: Any,
        projection: nn.Module,
    ) -> torch.Tensor:
        event_ptr = graph["event"].ptr
        anchors = states["event"][event_ptr[1:] - 1]
        means = global_mean_pool(states["event"], graph["event"].batch)
        return projection(torch.cat((anchors, means), dim=-1))

    def encode_partial_views(
        self, batch: dict[str, Any]
    ) -> tuple[
        dict[str, torch.Tensor],
        dict[str, dict[str, torch.Tensor]],
        torch.Tensor,
        dict[str, torch.Tensor],
    ]:
        graphs = batch["graphs"]
        missing = set(self.required_views) - set(graphs)
        if missing:
            raise ValueError(f"Missing Partial-L2 views: {sorted(missing)}")
        contexts: dict[str, torch.Tensor] = {}
        main_states_by_view: dict[str, dict[str, torch.Tensor]] = {}
        f80_shared = None
        for view in self.required_views:
            graph = graphs[view]
            edge_index_dict = dict(graph.edge_index_dict)
            shared = self._apply_layer(
                self._encode_nodes(graph),
                edge_index_dict,
                self.convolutions[0],
                self.norms[0],
                self.dropout,
            )
            main_states = self._apply_layer(
                shared,
                edge_index_dict,
                self.convolutions[1],
                self.norms[1],
                self.dropout,
            )
            contexts[view] = self._pool_context(
                main_states, graph, self.context_projection
            )
            main_states_by_view[view] = main_states
            if view == "f80":
                f80_shared = shared
        if f80_shared is None:
            raise RuntimeError("Partial-L2 requires the F80 view for Player")
        f80_graph = graphs["f80"]
        player_states = self._apply_layer(
            f80_shared,
            dict(f80_graph.edge_index_dict),
            self.player_convolution,
            self.player_norms,
            self.dropout,
        )
        player_context = self._pool_context(
            player_states, f80_graph, self.player_context_projection
        )
        return contexts, main_states_by_view, player_context, player_states

    def encode_partial_branches(
        self, batch: dict[str, Any]
    ) -> tuple[
        torch.Tensor,
        dict[str, torch.Tensor],
        torch.Tensor,
        dict[str, torch.Tensor],
    ]:
        """Compatibility wrapper for the original F80-only experiment."""

        contexts, states, player_context, player_states = self.encode_partial_views(batch)
        return contexts["f80"], states["f80"], player_context, player_states

    def forward_with_contexts(
        self, batch: dict[str, Any]
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        contexts, _, player_context, player_states = self.encode_partial_views(batch)
        task_context = self.task_contexts(contexts)
        graph = batch["graphs"]["f80"]
        player_batch = graph["player"].batch
        log_delta = F.softplus(self.time_head(task_context["time"]).squeeze(-1))
        predictions = {
            "event_logits": self.event_head(task_context["event"]),
            "log_delta": log_delta,
            "time_seconds": torch.expm1(log_delta).clamp(max=60.0),
            "position_xy": torch.sigmoid(self.position_head(task_context["position"])),
            "team_logits": self.team_actor_head(contexts["f80"]),
            "player_scores": self.player_actor_scorer(
                torch.cat(
                    (
                        player_context[player_batch],
                        player_states["player"],
                    ),
                    dim=-1,
                )
            ).squeeze(-1),
        }
        return predictions, {**contexts, "player": player_context}


class AgePoolingPartialL2HGT(PartialL2FiveTaskHGT):
    """One F80 HGT with shared or task-specific age-aware Event readout."""

    AGE_TASKS = ("event", "time", "position")

    def __init__(
        self,
        artifacts: ProtocolArtifacts,
        pooling_mode: str,
        dropout: float = 0.1,
    ) -> None:
        if pooling_mode not in {"shared", "task"}:
            raise ValueError(f"Unknown age pooling mode {pooling_mode!r}")
        super().__init__(artifacts, "five_f80", dropout)
        self.age_pooling_mode = pooling_mode

        # New readout parameters must not advance the RNG used by the common
        # backbone or subsequent training dropout. Shared and task variants
        # consume the same random draws and start from the same q0.
        with torch.random.fork_rng(devices=[]):
            self.age_embedding = nn.Embedding(80, 16)
            q0 = torch.empty(16)
            nn.init.normal_(q0, std=0.02)
            rows = 1 if pooling_mode == "shared" else len(self.AGE_TASKS)
            self.pooling_embeddings = nn.Parameter(q0.repeat(rows, 1))
            self.age_scorer = nn.Sequential(
                nn.Linear(32, 32),
                nn.GELU(),
                nn.Linear(32, 1),
            )
            nn.init.zeros_(self.age_scorer[-1].weight)
            nn.init.zeros_(self.age_scorer[-1].bias)

    @staticmethod
    def local_event_ages(graph: Any) -> torch.Tensor:
        ptr = graph["event"].ptr
        counts = ptr[1:] - ptr[:-1]
        local_position = torch.arange(
            int(ptr[-1]), device=ptr.device
        ) - torch.repeat_interleave(ptr[:-1], counts)
        return torch.repeat_interleave(counts - 1, counts) - local_position

    @classmethod
    def source_age_mismatch_count(cls, graph: Any) -> int:
        event = graph["event"]
        if not hasattr(event, "source_index"):
            raise ValueError("F80 Event nodes must expose source_index")
        ptr = event.ptr
        counts = ptr[1:] - ptr[:-1]
        anchor_source = event.source_index[ptr[1:] - 1]
        source_age = torch.repeat_interleave(anchor_source, counts) - event.source_index
        return int((source_age != cls.local_event_ages(graph)).sum().item())

    def _embedding_for_task(self, task: str, count: int) -> torch.Tensor:
        index = 0 if self.age_pooling_mode == "shared" else self.AGE_TASKS.index(task)
        return self.pooling_embeddings[index].unsqueeze(0).expand(count, -1)

    def raw_age_scores(self, task: str, ages: torch.Tensor) -> torch.Tensor:
        age_state = self.age_embedding(ages.long())
        query = self._embedding_for_task(task, int(ages.numel()))
        return self.age_scorer(torch.cat((age_state, query), dim=-1)).squeeze(-1)

    def age_weights(self, graph: Any, task: str) -> tuple[torch.Tensor, torch.Tensor]:
        ages = self.local_event_ages(graph)
        scores = self.raw_age_scores(task, ages)
        weights = pyg_softmax(
            scores,
            graph["event"].batch,
            num_nodes=int(graph["event"].ptr.numel() - 1),
        )
        return ages, weights

    def _age_context(
        self,
        states: dict[str, torch.Tensor],
        graph: Any,
        task: str,
    ) -> torch.Tensor:
        _, weights = self.age_weights(graph, task)
        event = graph["event"]
        anchors = states["event"][event.ptr[1:] - 1]
        pooled = global_add_pool(
            states["event"] * weights.unsqueeze(-1),
            event.batch,
            size=int(event.ptr.numel() - 1),
        )
        return self.context_projection(torch.cat((anchors, pooled), dim=-1))

    def forward_with_contexts(
        self, batch: dict[str, Any]
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        graph = batch["graphs"]["f80"]
        edges = dict(graph.edge_index_dict)
        shared = self._apply_layer(
            self._encode_nodes(graph), edges,
            self.convolutions[0], self.norms[0], self.dropout,
        )
        main_states = self._apply_layer(
            shared, edges,
            self.convolutions[1], self.norms[1], self.dropout,
        )
        mean_context = self._pool_context(
            main_states, graph, self.context_projection
        )
        if self.age_pooling_mode == "shared":
            shared_context = self._age_context(main_states, graph, "event")
            task_contexts = {task: shared_context for task in self.AGE_TASKS}
        else:
            task_contexts = {
                task: self._age_context(main_states, graph, task)
                for task in self.AGE_TASKS
            }

        player_states = self._apply_layer(
            shared, edges,
            self.player_convolution, self.player_norms, self.dropout,
        )
        player_context = self._pool_context(
            player_states, graph, self.player_context_projection
        )
        player_batch = graph["player"].batch
        log_delta = F.softplus(self.time_head(task_contexts["time"]).squeeze(-1))
        predictions = {
            "event_logits": self.event_head(task_contexts["event"]),
            "log_delta": log_delta,
            "time_seconds": torch.expm1(log_delta).clamp(max=60.0),
            "position_xy": torch.sigmoid(self.position_head(task_contexts["position"])),
            "team_logits": self.team_actor_head(mean_context),
            "player_scores": self.player_actor_scorer(
                torch.cat(
                    (player_context[player_batch], player_states["player"]), dim=-1
                )
            ).squeeze(-1),
        }
        return predictions, {
            "mean": mean_context,
            "player": player_context,
            **task_contexts,
        }


class AgePropagationPartialL2HGT(PartialL2FiveTaskHGT):
    """One standard F80 HGT plus low-rank task/age propagation residuals."""

    PROPAGATION_TASKS = PROPAGATION_TASKS

    def __init__(
        self,
        artifacts: ProtocolArtifacts,
        propagation_mode: str,
        dropout: float = 0.1,
    ) -> None:
        if propagation_mode not in PROPAGATION_MODES:
            raise ValueError(f"Unknown age propagation mode {propagation_mode!r}")
        super().__init__(artifacts, "five_f80", dropout)
        self.propagation_mode = propagation_mode
        # Consume no global RNG beyond the common Partial-L2 model. This keeps
        # training dropout and loader random streams aligned with the baseline.
        with torch.random.fork_rng(devices=[]):
            self.propagation_residuals = nn.ModuleList(
                LowRankRelationResidual(
                    tuple(POSSESSION_HGT_METADATA[0]),
                    tuple(POSSESSION_HGT_METADATA[1]),
                    rank=8,
                )
                for _ in range(2)
            )
            self.propagation_gate = (
                None
                if propagation_mode == "constant"
                else AgePropagationGate(propagation_mode, layers=2)
            )
        self.last_propagation_stats: dict[str, Any] = {}

    @staticmethod
    def source_age_mismatch_count(graph: Any) -> int:
        return source_age_mismatch_count(graph)

    def gate_profile(
        self, layer: int, task: str, ages: torch.Tensor
    ) -> torch.Tensor:
        if self.propagation_gate is None:
            return torch.ones_like(ages, dtype=torch.float32)
        return self.propagation_gate(layer, task, ages)

    def _residual_tasks(self) -> tuple[str, ...]:
        return (
            self.PROPAGATION_TASKS
            if self.propagation_mode == "task"
            else ("shared",)
        )

    def forward_with_contexts(
        self, batch: dict[str, Any]
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        graph = batch["graphs"]["f80"]
        edges = dict(graph.edge_index_dict)
        ages = local_event_ages(graph)
        base0 = self._encode_nodes(graph)
        base1 = self._apply_layer(
            base0, edges, self.convolutions[0], self.norms[0], self.dropout
        )
        base2 = self._apply_layer(
            base1, edges, self.convolutions[1], self.norms[1], self.dropout
        )
        player_states = self._apply_layer(
            base1, edges,
            self.player_convolution, self.player_norms, self.dropout,
        )

        task_states: dict[str, dict[str, torch.Tensor]] = {}
        propagation_stats: dict[str, Any] = {}
        capture_diagnostics = any(
            layer.capture_diagnostics for layer in self.propagation_residuals
        )
        for residual_task in self._residual_tasks():
            gate_task = "event" if residual_task == "shared" else residual_task
            delta1 = self.propagation_residuals[0](
                base0, edges, ages, self.propagation_gate, 0, gate_task
            )
            state1 = add_state_residual(base1, delta1)
            layer1_relations = (
                dict(self.propagation_residuals[0].last_relation_norms)
                if capture_diagnostics
                else {}
            )
            # state1 is deliberately used only by the lightweight residual;
            # Main HGT Layer 2 above always receives the shared base1 state.
            delta2 = self.propagation_residuals[1](
                state1, edges, ages, self.propagation_gate, 1, gate_task
            )
            task_states[residual_task] = add_state_residual(base2, delta2)
            if capture_diagnostics:
                propagation_stats[residual_task] = {
                    "layer1": {
                        "relation_message_norm": layer1_relations,
                        "node": self._residual_state_statistics(delta1, base1),
                    },
                    "layer2": {
                        "relation_message_norm": dict(
                            self.propagation_residuals[1].last_relation_norms
                        ),
                        "node": self._residual_state_statistics(delta2, base2),
                    },
                }
        if capture_diagnostics:
            self.last_propagation_stats = propagation_stats

        if self.propagation_mode == "task":
            task_contexts = {
                task: self._pool_context(
                    task_states[task], graph, self.context_projection
                )
                for task in self.PROPAGATION_TASKS
            }
        else:
            shared_context = self._pool_context(
                task_states["shared"], graph, self.context_projection
            )
            task_contexts = {
                task: shared_context for task in self.PROPAGATION_TASKS
            }

        mean_context = self._pool_context(base2, graph, self.context_projection)
        player_context = self._pool_context(
            player_states, graph, self.player_context_projection
        )
        player_batch = graph["player"].batch
        log_delta = F.softplus(self.time_head(task_contexts["time"]).squeeze(-1))
        predictions = {
            "event_logits": self.event_head(task_contexts["event"]),
            "log_delta": log_delta,
            "time_seconds": torch.expm1(log_delta).clamp(max=60.0),
            "position_xy": torch.sigmoid(
                self.position_head(task_contexts["position"])
            ),
            "team_logits": self.team_actor_head(mean_context),
            "player_scores": self.player_actor_scorer(
                torch.cat(
                    (player_context[player_batch], player_states["player"]), dim=-1
                )
            ).squeeze(-1),
        }
        return predictions, {
            "mean": mean_context,
            "player": player_context,
            **task_contexts,
        }

    @staticmethod
    def _residual_state_statistics(
        residual: dict[str, torch.Tensor], base: dict[str, torch.Tensor]
    ) -> dict[str, dict[str, float]]:
        values: dict[str, dict[str, float]] = {}
        for node_type in base:
            residual_state = residual[node_type].detach()
            base_state = base[node_type].detach()
            residual_norm = float(residual_state.norm().cpu())
            base_norm = float(base_state.norm().cpu())
            denominator = residual_norm * base_norm
            cosine = (
                0.0
                if denominator <= 1e-12
                else float(
                    ((residual_state * base_state).sum() / denominator).cpu()
                )
            )
            values[node_type] = {
                "residual_norm": residual_norm,
                "base_norm": base_norm,
                "norm_ratio": residual_norm / max(base_norm, 1e-12),
                "cosine": max(-1.0, min(1.0, cosine)),
            }
        return values


class ReceptiveFieldPartialL2HGT(PartialL2FiveTaskHGT):
    """Partial-L2 over one F80 graph with strict task propagation fields."""

    def __init__(self, artifacts: ProtocolArtifacts, configuration: str, dropout: float = 0.1) -> None:
        from .receptive_field import resolve_rf_spec

        super().__init__(artifacts, "five_f80", dropout)
        self.rf_configuration = configuration
        self.rf_spec = resolve_rf_spec(configuration)
        self._rf_forward_counts: dict[int, int] = {}

    def _encode_rf_nodes(
        self,
        graph: Any,
        base_states: dict[str, torch.Tensor],
        dynamic: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        states = dict(base_states)
        possession = graph["possession"]
        values = torch.cat(
            (
                (torch.log1p(dynamic["duration_so_far_seconds"]) / 5.0).unsqueeze(-1),
                (dynamic["event_count_so_far"] / 80.0).unsqueeze(-1),
                dynamic["current_position"],
                dynamic["current_position_mask"].float().unsqueeze(-1),
                dynamic["is_closed_as_of_anchor"].float().unsqueeze(-1),
                dynamic["is_current_active"].float().unsqueeze(-1),
                dynamic["dynamic_feature_mask"].float().unsqueeze(-1),
            ),
            dim=-1,
        )
        states["possession"] = self.possession_base.unsqueeze(0).expand(
            possession.num_nodes, -1
        ) + self.possession_dynamic_projection(values)
        return states

    @staticmethod
    def _filtered_edges(
        graph: Any, masks: dict[tuple[str, str, str], torch.Tensor]
    ) -> dict[tuple[str, str, str], torch.Tensor]:
        # Filtering happens before HGTConv, hence before attention softmax.
        return {
            edge_type: edge_index[:, masks[edge_type]]
            for edge_type, edge_index in graph.edge_index_dict.items()
        }

    @staticmethod
    def _pool_masked_context(
        states: dict[str, torch.Tensor], graph: Any, mask: torch.Tensor, projection: nn.Module
    ) -> torch.Tensor:
        anchors = states["event"][graph["event"].ptr[1:] - 1]
        means = global_mean_pool(
            states["event"][mask], graph["event"].batch[mask],
            size=int(graph["event"].ptr.numel() - 1),
        )
        return projection(torch.cat((anchors, means), dim=-1))

    def _rf_context(
        self,
        graph: Any,
        base_states: dict[str, torch.Tensor],
        metadata: dict[str, Any],
        count: int,
    ) -> torch.Tensor:
        self._rf_forward_counts[count] = self._rf_forward_counts.get(count, 0) + 1
        edges = self._filtered_edges(graph, metadata["edge_masks"])
        shared = self._apply_layer(
            self._encode_rf_nodes(graph, base_states, metadata["possession_dynamic"]),
            edges, self.convolutions[0], self.norms[0], self.dropout,
        )
        main = self._apply_layer(
            shared, edges, self.convolutions[1], self.norms[1], self.dropout,
        )
        return self._pool_masked_context(
            main, graph, metadata["event_pool_mask"], self.context_projection
        )

    def forward_with_contexts(
        self, batch: dict[str, Any]
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        graph = batch["graphs"]["f80"]
        full_edges = dict(graph.edge_index_dict)
        base_states = self._encode_nodes(graph)
        full_shared = self._apply_layer(
            base_states, full_edges,
            self.convolutions[0], self.norms[0], self.dropout,
        )
        full_main = self._apply_layer(
            full_shared, full_edges,
            self.convolutions[1], self.norms[1], self.dropout,
        )
        full_context = self._pool_context(full_main, graph, self.context_projection)
        player_states = self._apply_layer(
            full_shared, full_edges,
            self.player_convolution, self.player_norms, self.dropout,
        )
        player_context = self._pool_context(
            player_states, graph, self.player_context_projection
        )
        self._rf_forward_counts = {}
        rf_contexts = {
            count: self._rf_context(
                graph, base_states, batch["rf_metadata"][count], count
            )
            for count in self.rf_spec.unique_counts
        }
        task_context = {
            task: rf_contexts[count]
            for task, count in self.rf_spec.core_event_counts.items()
        }
        log_delta = F.softplus(self.time_head(task_context["time"]).squeeze(-1))
        player_batch = graph["player"].batch
        predictions = {
            "event_logits": self.event_head(task_context["event"]),
            "log_delta": log_delta,
            "time_seconds": torch.expm1(log_delta).clamp(max=60.0),
            "position_xy": torch.sigmoid(self.position_head(task_context["position"])),
            "team_logits": self.team_actor_head(full_context),
            "player_scores": self.player_actor_scorer(
                torch.cat((player_context[player_batch], player_states["player"]), dim=-1)
            ).squeeze(-1),
        }
        return predictions, {
            "f80": full_context,
            "player": player_context,
            **{f"n{count}": context for count, context in rf_contexts.items()},
        }


class StateAwarePartialL2FiveTaskHGT(PartialL2FiveTaskHGT):
    """Partial-L2 with sample-state scalar gates on concrete relations."""

    def __init__(
        self,
        artifacts: ProtocolArtifacts,
        dropout: float = 0.1,
    ) -> None:
        super().__init__(artifacts, "five_f80", dropout)
        shared_source = self.convolutions[0]
        main_source = self.convolutions[1]
        player_source = self.player_convolution
        self.convolutions[0] = StateAwareHGTConv.from_hgt(
            shared_source, POSSESSION_HGT_METADATA
        )
        self.convolutions[1] = StateAwareHGTConv.from_hgt(
            main_source, POSSESSION_HGT_METADATA
        )
        self.player_convolution = StateAwareHGTConv.from_hgt(
            player_source,
            POSSESSION_HGT_METADATA,
            controller_source=self.convolutions[1],
        )

    @staticmethod
    def _apply_state_aware_layer(
        states: dict[str, torch.Tensor],
        graph: Any,
        convolution: StateAwareHGTConv,
        norms: nn.ModuleDict,
        dropout: nn.Module,
        anchor_state: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        node_batches = {node_type: graph[node_type].batch for node_type in graph.node_types}
        updated = convolution(
            states,
            dict(graph.edge_index_dict),
            anchor_state,
            node_batches,
        )
        return {
            node_type: (
                norms[node_type](state + dropout(updated[node_type]))
                if node_type in updated
                else state
            )
            for node_type, state in states.items()
        }

    def encode_partial_views(
        self, batch: dict[str, Any]
    ) -> tuple[
        dict[str, torch.Tensor],
        dict[str, dict[str, torch.Tensor]],
        torch.Tensor,
        dict[str, torch.Tensor],
    ]:
        if set(batch["graphs"]) != {"f80"}:
            raise ValueError("State-aware gating experiment is F80-only")
        graph = batch["graphs"]["f80"]
        anchor_state = batch["anchor_state"]
        shared = self._apply_state_aware_layer(
            self._encode_nodes(graph),
            graph,
            self.convolutions[0],
            self.norms[0],
            self.dropout,
            anchor_state,
        )
        main_states = self._apply_state_aware_layer(
            shared,
            graph,
            self.convolutions[1],
            self.norms[1],
            self.dropout,
            anchor_state,
        )
        player_states = self._apply_state_aware_layer(
            shared,
            graph,
            self.player_convolution,
            self.player_norms,
            self.dropout,
            anchor_state,
        )
        main_context = self._pool_context(main_states, graph, self.context_projection)
        player_context = self._pool_context(
            player_states, graph, self.player_context_projection
        )
        return (
            {"f80": main_context},
            {"f80": main_states},
            player_context,
            player_states,
        )

    def gate_matrices(self) -> dict[str, torch.Tensor]:
        values = {
            "shared_l1": self.convolutions[0].last_gate_matrix,
            "main_l2": self.convolutions[1].last_gate_matrix,
            "player_l2": self.player_convolution.last_gate_matrix,
        }
        if any(value is None for value in values.values()):
            raise RuntimeError("Gate matrices are unavailable before a forward pass")
        return values  # type: ignore[return-value]


class PlayerHistoryEncoder(nn.Module):
    """Encode a candidate's K5 event sequence and causal prefix statistics."""

    def __init__(self) -> None:
        super().__init__()
        self.event_embedding = nn.Embedding(10, 16)
        self.token_projection = nn.Linear(23, 32)
        self.sequence_encoder = nn.GRU(32, 32, batch_first=True)
        self.statistics_encoder = nn.Sequential(
            nn.Linear(23, 32), nn.GELU()
        )
        self.output_projection = nn.Linear(64, 64)

    def forward(self, history: dict[str, torch.Tensor]) -> torch.Tensor:
        mask = history["sequence_mask"].bool()
        token = torch.cat(
            (
                self.event_embedding(history["event_type"].long()),
                history["numeric"].float(),
            ),
            dim=-1,
        )
        token = self.token_projection(token) * mask.unsqueeze(-1)
        sequence, _ = self.sequence_encoder(token)
        sequence_state = sequence[:, -1]
        seen = (history["lengths"] > 0).float().unsqueeze(-1)
        sequence_state = sequence_state * seen
        statistics_state = self.statistics_encoder(history["statistics"].float())
        return self.output_projection(
            torch.cat((sequence_state, statistics_state), dim=-1)
        )


class PlayerHistoryPartialL2HGT(PartialL2FiveTaskHGT):
    """Frozen low-level graph path with trainable private Player history branch."""

    def __init__(self, artifacts: ProtocolArtifacts, dropout: float = 0.1) -> None:
        super().__init__(artifacts, "five_f80", dropout)
        self.player_history_encoder = PlayerHistoryEncoder()
        self.player_history_adapter = nn.Sequential(
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 64),
        )
        nn.init.zeros_(self.player_history_adapter[-1].weight)
        nn.init.zeros_(self.player_history_adapter[-1].bias)

    def configure_trainable_private_branch(self) -> None:
        prefixes = (
            "player_convolution.",
            "player_norms.",
            "player_context_projection.",
            "player_actor_scorer.",
            "player_history_encoder.",
            "player_history_adapter.",
        )
        for name, parameter in self.named_parameters():
            parameter.requires_grad_(name.startswith(prefixes))

    def train(self, mode: bool = True) -> "PlayerHistoryPartialL2HGT":
        # Start with the whole inherited network in eval mode, then reactivate
        # only the private branch. This keeps frozen Projection dropout off.
        nn.Module.train(self, False)
        self.training = mode
        for module in (
            self.player_convolution,
            self.player_norms,
            self.player_context_projection,
            self.player_actor_scorer,
            self.player_history_encoder,
            self.player_history_adapter,
            self.dropout,
        ):
            module.train(mode)
        return self

    def _frozen_shared_states(self, graph: Any) -> dict[str, torch.Tensor]:
        with torch.no_grad():
            states = self._encode_nodes(graph)
            updated = self.convolutions[0](states, dict(graph.edge_index_dict))
            return {
                node_type: (
                    self.norms[0][node_type](state + updated[node_type])
                    if node_type in updated
                    else state
                ).detach()
                for node_type, state in states.items()
            }

    def forward_with_base_scores(
        self, batch: dict[str, Any]
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        graph = batch["graphs"]["f80"]
        shared = self._frozen_shared_states(graph)
        player_states = self._apply_layer(
            shared,
            dict(graph.edge_index_dict),
            self.player_convolution,
            self.player_norms,
            self.dropout,
        )
        player_context = self._pool_context(
            player_states, graph, self.player_context_projection
        )
        history_state = self.player_history_encoder(batch["player_history"])
        modified_player = player_states["player"] + self.player_history_adapter(
            history_state
        )
        player_batch = graph["player"].batch
        base_scores = self.player_actor_scorer(
            torch.cat((player_context[player_batch], player_states["player"]), dim=-1)
        ).squeeze(-1)
        scores = self.player_actor_scorer(
            torch.cat((player_context[player_batch], modified_player), dim=-1)
        ).squeeze(-1)
        return {"player_scores": scores}, base_scores

    def forward(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        predictions, _ = self.forward_with_base_scores(batch)
        return predictions


def build_target_model(
    artifacts: ProtocolArtifacts,
    task: str,
    method: str,
    joint_methods: dict[str, str] | None,
    *,
    graph_variant: str,
    possession_topology: str,
    possession_feature_level: str,
    dropout: float = 0.1,
) -> TargetStudyHGT:
    if graph_variant == "semantic_v2" or possession_topology == "none":
        return TargetStudyHGT(artifacts, task, method, joint_methods, dropout)
    if graph_variant != "semantic_v3_possession":
        raise ValueError(f"Unknown graph variant {graph_variant!r}")
    return PossessionTargetStudyHGT(
        artifacts,
        task,
        method,
        joint_methods,
        possession_feature_level,
        dropout,
    )


def build_multiview_model(
    artifacts: ProtocolArtifacts,
    mode: str,
    dropout: float = 0.1,
) -> TaskViewFusionHGT:
    return TaskViewFusionHGT(artifacts, mode, dropout)


def build_five_task_model(
    artifacts: ProtocolArtifacts,
    mode: str,
    dropout: float = 0.1,
    player_adapter: bool = False,
) -> FiveTaskViewHGT:
    return FiveTaskViewHGT(artifacts, mode, dropout, player_adapter)


def build_partial_l2_model(
    artifacts: ProtocolArtifacts,
    mode: str = "five_f80",
    dropout: float = 0.1,
) -> PartialL2FiveTaskHGT:
    return PartialL2FiveTaskHGT(artifacts, mode, dropout)


def build_player_history_model(
    artifacts: ProtocolArtifacts,
    dropout: float = 0.1,
) -> PlayerHistoryPartialL2HGT:
    return PlayerHistoryPartialL2HGT(artifacts, dropout)


def build_receptive_field_model(
    artifacts: ProtocolArtifacts,
    configuration: str,
    dropout: float = 0.1,
) -> ReceptiveFieldPartialL2HGT:
    return ReceptiveFieldPartialL2HGT(artifacts, configuration, dropout)


def build_age_pooling_model(
    artifacts: ProtocolArtifacts,
    pooling_mode: str,
    dropout: float = 0.1,
) -> AgePoolingPartialL2HGT:
    return AgePoolingPartialL2HGT(artifacts, pooling_mode, dropout)


def build_age_propagation_model(
    artifacts: ProtocolArtifacts,
    propagation_mode: str,
    dropout: float = 0.1,
) -> AgePropagationPartialL2HGT:
    return AgePropagationPartialL2HGT(artifacts, propagation_mode, dropout)


def build_state_aware_partial_l2_model(
    artifacts: ProtocolArtifacts,
    dropout: float = 0.1,
) -> StateAwarePartialL2FiveTaskHGT:
    return StateAwarePartialL2FiveTaskHGT(artifacts, dropout)
