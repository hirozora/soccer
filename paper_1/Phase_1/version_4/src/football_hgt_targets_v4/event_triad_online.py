"""Compose triad refinement without changing the four other deployed tasks."""

import time
import torch
from torch import nn

from football_benchmark.protocol import ProtocolArtifacts
from .constants import FEASIBILITY_ARTIFACT
from .event_posterior_online import load_integrated_model
from .event_triad import ROOT, TriadHead, correction_model, load_cache, predict, refine, state_features
from .five_task_training import FiveTaskTrainingConfig, _loader
from .position_head_refit import tensor_hash, write_json
from .training import _move_batch_to_device


class TriadIntegratedModel(nn.Module):
    def __init__(self, base, correction, mode):
        super().__init__()
        self.base = base.requires_grad_(False).eval()
        self.mode = mode
        if isinstance(correction, torch.Tensor):
            self.register_buffer("bias", correction)
            self.head = None
        else:
            self.head = correction
        self.eval()

    def train(self, mode=True):
        super().train(mode)
        self.base.eval()
        return self

    def forward(self, batch):
        captured = []
        handle = self.base.backbone.context_projection.register_forward_hook(lambda m, i, o: captured.append(o))
        try:
            with torch.no_grad():
                outputs = self.base(batch)
        finally:
            handle.remove()
        if len(captured) != 1:
            raise RuntimeError("Expected exactly one main context projection")
        if self.head is None:
            delta = self.bias
        else:
            graph = batch["graphs"]["f80"]
            anchors = graph["event"].ptr[1:] - 1
            vocabulary = torch.tensor(self.base.artifacts.event_type_ids, device=anchors.device)
            types = vocabulary[graph["event"].event_type_index[anchors]] - 1
            state = state_features(types, graph["event"].control_state_after_index[anchors])
            if self.mode == "context":
                state = torch.zeros_like(state)
            delta = self.head(captured[0], state)
        outputs["event_logits"] = refine(outputs["event_logits"], delta)
        return outputs


def load_model(seed, mode, root=ROOT, device="cpu"):
    return TriadIntegratedModel(load_integrated_model(seed, device=device),
                                correction_model(seed, mode, root, device), mode).to(device).requires_grad_(False).eval()


@torch.no_grad()
@torch.backends.mkldnn.flags(enabled=False)
def verify_online(seed, mode="original", root=ROOT, device="cpu"):
    cache, provenance = load_cache(seed, "validation", root)
    base = load_integrated_model(seed, device=device)
    candidate = load_model(seed, mode, root, device)
    before = tensor_hash(base.backbone.state_dict())
    if before != tensor_hash(candidate.base.backbone.state_dict()):
        raise RuntimeError("Frozen baseline differs")
    counts = {k: 0 for k in ("encoder", "shared_l1", "main_l2", "player_l2")}
    def hook(key):
        def count(*unused):
            counts[key] += 1
        return count
    b = candidate.base.backbone
    handles = [b.event_projection.register_forward_hook(hook("encoder")),
               b.convolutions[0].register_forward_hook(hook("shared_l1")),
               b.convolutions[1].register_forward_hook(hook("main_l2")),
               b.player_convolution.register_forward_hook(hook("player_l2"))]
    config = FiveTaskTrainingConfig(mode="five_f80", output_dir=root, seed=seed, device=device,
        num_workers=0, max_validation_samples=32, batch_size=16)
    loader = _loader("validation", config, ProtocolArtifacts.load(FEASIBILITY_ARTIFACT), False)
    row_map = {s: i for i, s in enumerate(cache["sample_ids"])}
    errors = {key: 0. for key in ("time_seconds", "log_delta", "position_xy", "team_logits", "player_scores")}
    cache_error, probability_error, batches = 0., 0., 0
    timings = {"original": [], "candidate": []}
    for raw in loader:
        rows = torch.tensor([row_map[s] for s in raw["sample_ids"]])
        if (not torch.equal(raw["targets"]["raw_event_10"], cache["event_true"][rows])
                or not torch.equal(raw["current_event_indices"], cache["current_event_indices"][rows])):
            raise RuntimeError("Online/cache alignment mismatch")
        batch = _move_batch_to_device(raw, torch.device(device))
        pair = []
        for name, model in (("original", base), ("candidate", candidate)):
            started = time.perf_counter()
            pair.append(model(batch))
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            timings[name].append(time.perf_counter() - started)
        old, new = pair
        for key in errors:
            errors[key] = max(errors[key], float((old[key] - new[key]).abs().max()))
        subset = {k: v[rows] for k, v in cache.items() if isinstance(v, torch.Tensor)}
        expected = (refine(subset["base_event_logits"], candidate.bias.cpu()) if candidate.head is None
                    else predict(candidate.head, subset, mode))
        cache_error = max(cache_error, float((new["event_logits"].cpu() - expected).abs().max()))
        probability_error = max(probability_error, float((new["event_logits"].cpu().softmax(-1) - expected.softmax(-1)).abs().max()))
        if not torch.equal(new["event_logits"].cpu().argmax(-1), expected.argmax(-1)):
            raise RuntimeError("Online/cache predictions differ")
        batches += 1
    for handle in handles:
        handle.remove()
    if (not batches or max([probability_error, *errors.values()]) >= 1e-6
            or any(n != batches for n in counts.values()) or before != tensor_hash(candidate.base.backbone.state_dict())
            or any(p.grad is not None for p in candidate.parameters())):
        raise RuntimeError(f"Online verification failed: {errors}, {counts}, {probability_error}")
    result = {"passed": True, "seed": seed, "mode": mode, "backbone_hash": before,
              "unaffected_output_max_errors": errors, "layer_calls": counts,
              "cache_logit_max_error": cache_error, "cache_probability_max_error": probability_error,
              "provenance": provenance, "descriptive_model_timings_seconds": timings}
    write_json(root / "verification" / f"{mode}_seed{seed}.json", result)
    return result
