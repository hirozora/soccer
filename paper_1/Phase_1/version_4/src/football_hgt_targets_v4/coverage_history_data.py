"""Deterministic supervision rotation and strictly lagged match-history priors."""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from football_benchmark.data import CanonicalEventDataset, load_records
from football_benchmark.sampling import TargetSamplePlan
from .constants import SAMPLE_PLAN, POSSESSION_GRAPH_ROOT
from .oracle_dependency import load_roster_team_map


SELECTOR_SEED = 20260923
PHASE = Path(__file__).resolve().parents[3]
MATCHES = PHASE / "data/whyscout/raw/matches/matches_England.json"


def stable_seed(*parts):
    return int.from_bytes(hashlib.sha256(":".join(map(str, parts)).encode()).digest()[:8], "little")


def rotation_table(num_transitions, first, match_id, epochs=24):
    first = tuple(first)
    count = len(first)
    if count != min(128, num_transitions) or sorted(set(first)) != list(first):
        raise ValueError("Invalid original stratified plan")
    table = torch.empty((epochs, count), dtype=torch.long)
    for slot, initial in enumerate(first):
        lo, hi = slot * num_transitions // count, (slot + 1) * num_transitions // count
        if not lo <= initial < hi:
            raise ValueError("Original target outside its stratum")
        remaining = [i for i in range(lo, hi) if i != initial]
        rng = np.random.default_rng(stable_seed(SELECTOR_SEED, match_id, slot))
        order = [initial, *rng.permutation(remaining).tolist()]
        table[:, slot] = torch.tensor([order[e % len(order)] for e in range(epochs)])
    return table


class RotatingEventDataset(CanonicalEventDataset):
    """Workers see epoch changes through shared memory, not parent-only lists."""

    def __init__(self, records, artifacts, *, rotate, max_samples=None):
        plan = TargetSamplePlan.load(SAMPLE_PLAN).currents_by_match("train")
        super().__init__(records, artifacts, 80, max_samples, plan)
        self.rotate = rotate
        self.epoch_state = torch.ones(1, dtype=torch.long).share_memory_()
        self.tables = [rotation_table(r.num_events - 1, plan[r.match_id], r.match_id).share_memory_()
                       for r in records]
        self.local_epoch = None

    def set_epoch(self, epoch):
        if not 1 <= epoch <= 24:
            raise ValueError("Epoch outside fixed budget")
        self.epoch_state[0] = epoch

    def __getitem__(self, index):
        epoch = int(self.epoch_state[0]) if self.rotate else 1
        if self.local_epoch != epoch:
            self.currents = [tuple(t[epoch - 1].tolist()) for t in self.tables]
            self.local_epoch = epoch
        return super().__getitem__(index)

    def plan_hash(self, epoch):
        e = epoch - 1 if self.rotate else 0
        return hashlib.sha256(torch.cat([t[e] for t in self.tables]).numpy().tobytes()).hexdigest()


def summary13(counts, games, global_counts):
    counts = np.asarray(counts, dtype=np.float64)
    total = counts.sum()
    if total == 0:
        return np.zeros(13, dtype=np.float32)
    global_counts = np.asarray(global_counts, dtype=np.float64)
    prior = global_counts / global_counts.sum() if global_counts.sum() else np.full(10, .1)
    return np.array([*((counts + 20 * prior) / (total + 20)), np.log1p(total), np.log1p(games), 1], dtype=np.float32)


def derangement(mapping, match_id):
    groups = defaultdict(list)
    for player, team in mapping.items():
        if player > 0:
            groups[team].append(player)
    result = {}
    for team, players in groups.items():
        players.sort()
        if len(players) < 2:
            continue
        offset = 1 + stable_seed(SELECTOR_SEED, match_id, team) % (len(players) - 1)
        result.update({p: players[(i + offset) % len(players)] for i, p in enumerate(players)})
    return result


def utc(value):
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed.astimezone(timezone.utc)


class MatchHistoryIndex:
    """Only allowed splits are loaded; completed matches enter after a 24h lag."""

    def __init__(self, include_test=False, root=None):
        if include_test:
            from .coverage_history_training import require_test
            if root is None:
                raise RuntimeError("Test history requires an explicit locked experiment")
            require_test(root)
        self.roster = load_roster_team_map()
        metadata = {int(m["wyId"]): m for m in json.loads(MATCHES.read_text())}
        records = [r for s in (("train", "validation", "test") if include_test else ("train", "validation"))
                   for r in load_records(s, graph_root=POSSESSION_GRAPH_ROOT)]
        records.sort(key=lambda r: (utc(metadata[r.match_id]["dateutc"]), r.match_id))
        team_counts = defaultdict(lambda: np.zeros(10, dtype=np.int64))
        player_counts = defaultdict(lambda: np.zeros(10, dtype=np.int64))
        team_games, player_games = defaultdict(int), defaultdict(int)
        global_counts = np.zeros(10, dtype=np.int64)
        cursor, sources = 0, []
        self.snapshots = {}
        self.offside_audit = {"events": 0, "unknown_player": 0, "roster_team_mismatch": 0,
                              "interpretation": "Recorded actor only; offside-winning defensive team is not inferred."}

        def match_summary(record):
            graph = torch.load(record.graph_path, map_location="cpu", weights_only=False)
            event = graph["node_stores"]["event"]
            teams = graph["node_stores"]["team"]["raw_id"][event["team_local_index"]].tolist()
            players = graph["node_stores"]["player"]["raw_id"][event["player_local_index"]].tolist()
            types = event["event_type_index"].tolist()
            tc, pc = defaultdict(lambda: np.zeros(10, dtype=np.int64)), defaultdict(lambda: np.zeros(10, dtype=np.int64))
            for t, p, c in zip(teams, players, types):
                tc[t][c] += 1
                if p > 0:
                    pc[p][c] += 1
                if c == 5:
                    self.offside_audit["events"] += 1
                    self.offside_audit["unknown_player"] += int(p <= 0)
                    mapped = self.roster[record.match_id].get(p)
                    self.offside_audit["roster_team_mismatch"] += int(mapped is not None and mapped != t)
            return tc, pc, np.bincount(types, minlength=10)

        for target in records:
            cutoff = utc(metadata[target.match_id]["dateutc"]) - timedelta(hours=24)
            while cursor < len(records) and utc(metadata[records[cursor].match_id]["dateutc"]) <= cutoff:
                source = records[cursor]
                if metadata[source.match_id]["status"] != "Played":
                    raise RuntimeError("Historical match not completed")
                tc, pc, gc = match_summary(source)
                for t, counts in tc.items():
                    team_counts[t] += counts
                    team_games[t] += 1
                for p, counts in pc.items():
                    player_counts[p] += counts
                    player_games[p] += 1
                global_counts += gc
                sources.append(source.match_id)
                cursor += 1
            teams = sorted(map(int, metadata[target.match_id]["teamsData"]))
            if len(teams) != 2:
                raise RuntimeError("Expected two match teams")
            roster = self.roster[target.match_id]
            shuffle = derangement(roster, target.match_id)
            self.snapshots[target.match_id] = {
                "cutoff": cutoff.isoformat(), "source_matches": tuple(sources), "teams": teams,
                "team": {t: summary13(team_counts[t], team_games[t], global_counts) for t in teams},
                "player": {p: summary13(player_counts[p], player_games[p], global_counts) for p in roster},
                "team_games": {t: team_games[t] for t in teams},
                "player_games": {p: player_games[p] for p in roster}, "shuffle": shuffle,
            }

    def candidate_features(self, match_id, anchor_team, candidates):
        snapshot = self.snapshots[int(match_id)]
        if int(anchor_team) not in snapshot["teams"]:
            raise RuntimeError("Anchor team not in match roster")
        opponent = next(t for t in snapshot["teams"] if t != int(anchor_team))
        teams = np.concatenate([snapshot["team"][int(anchor_team)], snapshot["team"][opponent]])
        correct = np.zeros((len(candidates), 13), dtype=np.float32)
        shuffled = np.zeros_like(correct)
        for i, p in enumerate(candidates):
            other = snapshot["shuffle"].get(int(p))
            if other is not None:
                correct[i] = snapshot["player"][int(p)]
                shuffled[i] = snapshot["player"][other]
        return teams, correct, shuffled


def condition39(index, match_ids, anchors, candidates, ptr, scores):
    """No targets/masks are accepted by this API."""
    correct, shuffled, groups = [], [], []
    for row, match_id in enumerate(match_ids):
        lo, hi = int(ptr[row]), int(ptr[row + 1])
        teams, own, wrong = index.candidate_features(int(match_id), int(anchors[row]), candidates[lo:hi])
        weight = scores[lo:hi].detach().cpu().softmax(0).numpy()
        correct.append(np.concatenate([teams, weight @ own]))
        shuffled.append(np.concatenate([teams, weight @ wrong]))
        snap = index.snapshots[int(match_id)]
        games = snap["team_games"][int(anchors[row])]
        eligible = np.array([int(p) in snap["shuffle"] for p in candidates[lo:hi]])
        groups.append([games, float(weight @ own[:, -1]), float(eligible.mean())])
    return torch.from_numpy(np.stack(correct)), torch.from_numpy(np.stack(shuffled)), torch.tensor(groups)
