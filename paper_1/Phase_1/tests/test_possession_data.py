from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from football_hgt.event_tables import build_event_tables
from football_hgt.possession import (
    DEFAULT_RULES_PATH,
    infer_match_possessions,
    load_possession_rules,
    validate_possessions,
)


class EventTableBuilderTest(unittest.TestCase):
    def test_normalized_tables_preserve_order_positions_and_all_tags(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "data"
            output = Path(temporary) / "out"
            for directory in (
                root / "raw/events",
                root / "raw/matches",
                root / "raw/mappings",
                root / "raw/entities",
            ):
                directory.mkdir(parents=True)
            (root / "raw/mappings/eventid2name.csv").write_text(
                "event,subevent,event_label,subevent_label\n"
                "1,10,Duel,Air duel\n"
                "8,85,Pass,Simple pass\n",
                encoding="utf-8",
            )
            (root / "raw/mappings/tags2name.csv").write_text(
                "Tag,Label,Description\n"
                "702,neutral,Neutral\n"
                "1801,accurate,Accurate\n",
                encoding="utf-8",
            )
            (root / "raw/entities/players.json").write_text(
                json.dumps([{"wyId": 11}]), encoding="utf-8"
            )
            match = {
                "wyId": 100,
                "competitionId": 364,
                "seasonId": 1,
                "roundId": 1,
                "gameweek": 1,
                "date": "2018-01-01",
                "dateutc": "2018-01-01 12:00:00",
                "duration": "Regular",
                "status": "Played",
                "winner": 1,
                "venue": "Test",
                "label": "A - B",
                "teamsData": {
                    "1": {"teamId": 1, "side": "home", "score": 1},
                    "2": {"teamId": 2, "side": "away", "score": 0},
                },
            }
            (root / "raw/matches/matches_England.json").write_text(
                json.dumps([match]), encoding="utf-8"
            )
            events = [
                {
                    "eventId": 8,
                    "subEventId": 85,
                    "eventName": "Pass",
                    "subEventName": "Simple pass",
                    "eventSec": 5.0,
                    "id": 2,
                    "matchId": 100,
                    "matchPeriod": "2H",
                    "playerId": 11,
                    "teamId": 2,
                    "positions": [{"x": 50, "y": 40}, {"x": 80, "y": 40}],
                    "tags": [{"id": 1801}],
                },
                {
                    "eventId": 1,
                    "subEventId": 10,
                    "eventName": "Duel",
                    "subEventName": "Air duel",
                    "eventSec": 10.0,
                    "id": 1,
                    "matchId": 100,
                    "matchPeriod": "1H",
                    "playerId": 0,
                    "teamId": 1,
                    "positions": [{"x": 20, "y": 30}],
                    "tags": [{"id": 702}],
                },
            ]
            (root / "raw/events/events_England.json").write_text(
                json.dumps(events), encoding="utf-8"
            )

            manifest = build_event_tables(root, output, "England")
            normalized = pd.read_parquet(output / "England/events.parquet")
            tags = pd.read_parquet(output / "England/event_tags.parquet")

            self.assertEqual(manifest["totals"]["events"], 2)
            self.assertEqual(normalized["event_uid"].tolist(), [1, 2])
            self.assertEqual(normalized["source_event_index"].tolist(), [1, 0])
            self.assertEqual(normalized["event_index"].tolist(), [0, 1])
            self.assertFalse(bool(normalized.iloc[0].end_position_valid))
            self.assertAlmostEqual(float(normalized.iloc[1].end_x), 0.8)
            self.assertEqual(tags["tag_id"].tolist(), [702, 1801])
            self.assertEqual(int(normalized["tag_count"].sum()), len(tags))


class CausalPossessionInferenceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rules = load_possession_rules(DEFAULT_RULES_PATH)

    def _events(self) -> pd.DataFrame:
        definitions = [
            # team, event, subevent, period, seconds
            (1, 8, 85, "Pass", "Simple pass", "1H", 1.0),
            (2, 1, 12, "Duel", "Ground defending duel", "1H", 2.0),
            (2, 8, 85, "Pass", "Simple pass", "1H", 3.0),
            (2, 10, 100, "Shot", "Shot", "1H", 4.0),
            (1, 2, 20, "Foul", "Foul", "1H", 5.0),
            (1, 3, 36, "Free Kick", "Throw in", "1H", 6.0),
            (1, 8, 85, "Pass", "Simple pass", "1H", 7.0),
            (1, 8, 85, "Pass", "Simple pass", "1H", 8.0),
            (2, 8, 85, "Pass", "Simple pass", "2H", 2705.0),
        ]
        rows = []
        for index, (team, event_id, subevent_id, event_name, subevent_name, period, seconds) in enumerate(definitions):
            rows.append(
                {
                    "match_id": 1,
                    "event_uid": 100 + index,
                    "event_index": index,
                    "event_type_id": event_id,
                    "event_name": event_name,
                    "subevent_id": subevent_id,
                    "subevent_name": subevent_name,
                    "period": period,
                    "absolute_seconds": seconds,
                    "team_id": team,
                    "start_x": 0.1 * index,
                    "start_y": 0.5,
                    "start_position_valid": True,
                    "end_x": 0.1 * index + 0.05,
                    "end_y": 0.5,
                    "end_position_valid": True,
                }
            )
        return pd.DataFrame(rows)

    def _tags(self) -> pd.DataFrame:
        tags = {
            100: [1801],
            101: [703],
            102: [1801],
            103: [101],
            106: [1802],
            107: [1801],
            108: [1801],
        }
        rows = []
        for event_uid, values in tags.items():
            for order, tag_id in enumerate(values):
                rows.append(
                    {
                        "match_id": 1,
                        "event_uid": event_uid,
                        "event_index": event_uid - 100,
                        "tag_order": order,
                        "tag_id": tag_id,
                    }
                )
        return pd.DataFrame(rows)

    def test_contest_does_not_switch_and_control_reanchors(self) -> None:
        events = self._events()
        tags_by_event = {
            int(event_uid): set(int(value) for value in group.tag_id)
            for event_uid, group in self._tags().groupby("event_uid")
        }
        states, possessions = infer_match_possessions(
            events, tags_by_event, (1, 2), self.rules
        )

        first_uid = states.iloc[0].possession_uid
        self.assertEqual(states.iloc[1].possession_uid, first_uid)
        self.assertEqual(states.iloc[1].owner_team_id, 1)
        self.assertEqual(states.iloc[1].state_after, "CONTESTED")
        self.assertFalse(bool(states.iloc[1].switch_confirmed))

        self.assertNotEqual(states.iloc[2].possession_uid, first_uid)
        self.assertEqual(states.iloc[2].owner_team_id, 2)
        self.assertTrue(bool(states.iloc[2].switch_confirmed))
        self.assertTrue(bool(states.iloc[3].is_closed_as_of_event))
        self.assertEqual(states.iloc[3].boundary_reason, "tag_confirmed_close")

        self.assertEqual(states.iloc[6].state_after, "CONTESTED")
        self.assertEqual(states.iloc[7].state_after, "CONTROL")
        self.assertEqual(states.iloc[7].owner_team_id, 1)
        self.assertFalse(bool(states.iloc[8].switch_confirmed))
        self.assertEqual(states.iloc[8].owner_team_id, 2)
        self.assertEqual(states.iloc[8].boundary_reason, "period_break")
        self.assertFalse(bool(possessions.iloc[-1].is_closed_audit_only))

    def test_candidates_are_evidence_only(self) -> None:
        events = self._events()
        tags_by_event = {
            int(event_uid): set(int(value) for value in group.tag_id)
            for event_uid, group in self._tags().groupby("event_uid")
        }
        states, _ = infer_match_possessions(
            events, tags_by_event, (1, 2), self.rules
        )

        # Team 2 won the duel, but the event remains in Team 1's possession.
        self.assertEqual(int(states.iloc[1].candidate_team_id), 2)
        self.assertEqual(int(states.iloc[1].owner_team_id), 1)
        self.assertFalse(bool(states.iloc[1].switch_confirmed))

        # Team 1's inaccurate pass only nominates Team 2 as the candidate.
        self.assertEqual(int(states.iloc[6].candidate_team_id), 2)
        self.assertEqual(int(states.iloc[6].owner_team_id), 1)
        self.assertFalse(bool(states.iloc[6].switch_confirmed))

    def test_prefix_is_identical_to_full_stream(self) -> None:
        events = self._events()
        tags_by_event = {
            int(event_uid): set(int(value) for value in group.tag_id)
            for event_uid, group in self._tags().groupby("event_uid")
        }
        full, _ = infer_match_possessions(events, tags_by_event, (1, 2), self.rules)
        for cutoff in range(1, len(events) + 1):
            prefix, _ = infer_match_possessions(
                events.iloc[:cutoff], tags_by_event, (1, 2), self.rules
            )
            self.assertTrue(
                prefix.reset_index(drop=True).equals(
                    full.iloc[:cutoff].reset_index(drop=True)
                ),
                msg=f"prefix mismatch at cutoff {cutoff}",
            )

        repeated, _ = infer_match_possessions(
            events.copy(), tags_by_event, (1, 2), self.rules
        )
        self.assertTrue(full.equals(repeated))

    def test_validator_accepts_causal_fixture(self) -> None:
        events = self._events()
        tags = self._tags()
        tags_by_event = {
            int(event_uid): set(int(value) for value in group.tag_id)
            for event_uid, group in tags.groupby("event_uid")
        }
        states, possessions = infer_match_possessions(
            events, tags_by_event, (1, 2), self.rules
        )
        match_teams = pd.DataFrame(
            [{"match_id": 1, "team_id": 1}, {"match_id": 1, "team_id": 2}]
        )
        report = validate_possessions(
            events,
            tags,
            match_teams,
            states,
            possessions,
            self.rules,
            prefix_checks=10,
        )
        self.assertTrue(report["valid"], report["errors"])


if __name__ == "__main__":
    unittest.main()
