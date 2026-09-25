from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch

from football_hgt.catalog import load_catalog
from football_hgt.dataset import (
    FixedWindowDataset,
    MatchGraphRecord,
    load_match_graph,
    sample_fixed_event_window,
    slice_event_prefix,
    validate_fixed_event_window,
)
from football_hgt.graph_builder import build_match_graph, validate_graph
from football_hgt.schema import RESULT_DIRECTION_TO_INDEX, classify_result_direction


PHASE_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PHASE_ROOT / "data/whyscout"


class GraphBuilderTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.catalog = load_catalog(DATA_ROOT)
        match_path = DATA_ROOT / "raw/matches/matches_World_Cup.json"
        cls.match = json.loads(match_path.read_text(encoding="utf-8"))[0]
        cls.match_id = int(cls.match["wyId"])
        cls.competition_id = int(cls.match["competitionId"])
        cls.team_ids = [int(value) for value in cls.match["teamsData"]]
        cls.player_id = next(iter(cls.catalog.players))

    def _events(self) -> list[dict]:
        return [
            {
                "eventId": 1,
                "subEventName": "",
                "tags": [{"id": 702}],
                "playerId": 0,
                "positions": [{"x": 20, "y": 30}],
                "matchId": self.match_id,
                "eventName": "Duel",
                "teamId": self.team_ids[0],
                "matchPeriod": "1H",
                "eventSec": 2800.0,
                "subEventId": "",
                "id": 1,
            },
            {
                "eventId": 8,
                "subEventName": "Simple pass",
                "tags": [{"id": 1801}],
                "playerId": self.player_id,
                "positions": [{"x": 50, "y": 50}, {"x": 75, "y": 40}],
                "matchId": self.match_id,
                "eventName": "Pass",
                "teamId": self.team_ids[1],
                "matchPeriod": "2H",
                "eventSec": 5.0,
                "subEventId": 85,
                "id": 2,
            },
            {
                "eventId": 10,
                "subEventName": "Shot",
                "tags": [{"id": 101}],
                "playerId": self.player_id,
                "positions": [{"x": 90, "y": 50}, {"x": 100, "y": 50}],
                "matchId": self.match_id,
                "eventName": "Shot",
                "teamId": self.team_ids[1],
                "matchPeriod": "2H",
                "eventSec": 8.0,
                "subEventId": 100,
                "id": 3,
            },
        ]

    def test_build_match_graph_contract(self) -> None:
        graph = build_match_graph(
            self.match, self._events(), self.catalog, "World_Cup"
        )
        self.assertEqual(validate_graph(graph), [])

        event_store = graph["node_stores"]["event"]
        self.assertEqual(event_store["num_nodes"], 3)
        self.assertTrue(
            torch.allclose(
                event_store["absolute_seconds"],
                torch.tensor([2800.0, 2805.0, 2808.0]),
            )
        )
        self.assertEqual(event_store["subevent_type_index"][0].item(), 0)
        self.assertEqual(event_store["end_position_mask"].tolist(), [False, True, True])
        self.assertEqual(event_store["retained_tag_count"].tolist(), [0, 1, 1])

        tag_edges = graph["edge_stores"]["event__has_tag__tag"]["edge_index"]
        self.assertEqual(tag_edges.shape[1], 2)
        self.assertNotIn(0, tag_edges[0].tolist())

        self.assertEqual(graph["targets"]["delta_seconds"].tolist(), [5.0, 3.0, 0.0])
        self.assertEqual(
            graph["targets"]["result_direction"].tolist(),
            [
                RESULT_DIRECTION_TO_INDEX["favorable"],
                RESULT_DIRECTION_TO_INDEX["favorable"],
                -1,
            ],
        )
        self.assertEqual(graph["targets"]["player_known_mask"].tolist(), [True, True, False])

    def test_prefix_slice_is_causal_and_keeps_current_target(self) -> None:
        graph = build_match_graph(
            self.match, self._events(), self.catalog, "World_Cup"
        )
        prefix = slice_event_prefix(graph, end_event_index=1)
        self.assertEqual(prefix["node_stores"]["event"]["num_nodes"], 2)
        self.assertEqual(prefix["targets"]["mask"].tolist(), [True, True])
        self.assertEqual(
            prefix["edge_stores"]["event__next__event"]["edge_index"].tolist(),
            [[0], [1]],
        )
        for edge_type, store in prefix["edge_stores"].items():
            source, _, destination = edge_type.split("__")
            edge_index = store["edge_index"]
            if source == "event" and edge_index.numel():
                self.assertLess(int(edge_index[0].max()), 2)
            if destination == "event" and edge_index.numel():
                self.assertLess(int(edge_index[1].max()), 2)

    def test_serialization_round_trip(self) -> None:
        graph = build_match_graph(
            self.match, self._events(), self.catalog, "World_Cup"
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "graph.pt"
            torch.save(graph, path)
            loaded = load_match_graph(path)
        self.assertEqual(loaded["match_id"], self.match_id)
        self.assertTrue(
            torch.equal(
                loaded["node_stores"]["event"]["raw_id"],
                graph["node_stores"]["event"]["raw_id"],
            )
        )

    def test_fixed_window_is_causal_relabelled_and_training_ready(self) -> None:
        graph = build_match_graph(
            self.match, self._events(), self.catalog, "World_Cup"
        )
        original_event_ids = graph["node_stores"]["event"]["raw_id"].clone()
        window = sample_fixed_event_window(
            graph, current_event_index=1, window_size=1, validate=True
        )

        self.assertEqual(validate_fixed_event_window(window), [])
        self.assertEqual(window["window"]["start_event_index"], 1)
        self.assertEqual(window["window"]["target_event_index"], 2)
        self.assertEqual(window["window"]["query_event_index"], 0)
        self.assertEqual(window["node_stores"]["event"]["raw_id"].tolist(), [2])
        self.assertNotIn(
            3, window["node_stores"]["event"]["raw_id"].tolist()
        )
        self.assertEqual(
            window["edge_stores"]["event__next__event"]["edge_index"].shape,
            (2, 0),
        )
        for store in window["edge_stores"].values():
            edge_index = store["edge_index"]
            if edge_index.numel():
                self.assertLess(int(edge_index[1].max()), 1)

        targets = window["targets"]
        self.assertEqual(targets["event_type_index"].item(), 9)
        self.assertAlmostEqual(targets["delta_seconds"].item(), 3.0)
        self.assertAlmostEqual(
            targets["log_delta_seconds"].item(), torch.log1p(torch.tensor(3.0)).item()
        )
        self.assertEqual(targets["acting_side_index"].item(), 1)
        self.assertTrue(targets["player_mask"].item())
        self.assertEqual(targets["advantage_index"].item(), 1)
        self.assertTrue(targets["advantage_mask"].item())

        self.assertTrue(
            torch.equal(graph["node_stores"]["event"]["raw_id"], original_event_ids)
        )
        self.assertEqual(
            window["node_stores"]["event"]["raw_id"].untyped_storage().data_ptr(),
            graph["node_stores"]["event"]["raw_id"].untyped_storage().data_ptr(),
        )
        self.assertIs(
            window["node_stores"]["player"]["raw_id"],
            graph["node_stores"]["player"]["raw_id"],
        )

    def test_fixed_window_dataset_maps_global_indices_to_matches(self) -> None:
        graph = build_match_graph(
            self.match, self._events(), self.catalog, "World_Cup"
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "graph.pt"
            torch.save(graph, path)
            dataset = FixedWindowDataset(
                [
                    MatchGraphRecord(
                        competition_slug="World_Cup",
                        match_id=self.match_id,
                        graph_path=path,
                        num_events=3,
                    )
                ],
                window_size=2,
                validate_graph_on_load=True,
                validate_samples=True,
            )
            self.assertEqual(len(dataset), 2)
            first = dataset[0]
            second = dataset[-1]

        self.assertEqual(first["window"]["current_event_index"], 0)
        self.assertEqual(first["window"]["num_events"], 1)
        self.assertEqual(second["window"]["current_event_index"], 1)
        self.assertEqual(second["window"]["num_events"], 2)

    def test_result_direction_priority(self) -> None:
        self.assertEqual(
            classify_result_direction(
                "Shot", {101, 1801}, self.catalog.result_direction_rules
            ),
            RESULT_DIRECTION_TO_INDEX["favorable"],
        )
        self.assertEqual(
            classify_result_direction(
                "Shot", {101, 2101}, self.catalog.result_direction_rules
            ),
            RESULT_DIRECTION_TO_INDEX["favorable"],
        )
        self.assertEqual(
            classify_result_direction(
                "Save attempt", {101, 1802}, self.catalog.result_direction_rules
            ),
            RESULT_DIRECTION_TO_INDEX["unfavorable"],
        )
        self.assertEqual(
            classify_result_direction(
                "Pass", {1401, 1801}, self.catalog.result_direction_rules
            ),
            RESULT_DIRECTION_TO_INDEX["unfavorable"],
        )
        self.assertEqual(
            classify_result_direction(
                "Duel", {703, 1802}, self.catalog.result_direction_rules
            ),
            RESULT_DIRECTION_TO_INDEX["favorable"],
        )
        self.assertEqual(
            classify_result_direction(
                "Duel", {702}, self.catalog.result_direction_rules
            ),
            RESULT_DIRECTION_TO_INDEX["neutral_or_unknown"],
        )


if __name__ == "__main__":
    unittest.main()
