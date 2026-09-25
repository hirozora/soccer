from __future__ import annotations

import unittest

import pandas as pd

from football_hgt.possession import infer_match_possessions
from football_hgt.possession_v2 import (
    DEFAULT_CANDIDATE_RULES_PATH_V2,
    DEFAULT_RULES_PATH_V2,
    infer_match_possessions_v2,
    load_possession_rules_v2,
    validate_possessions_v2,
)


class CausalPossessionV2Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rules = load_possession_rules_v2(
            DEFAULT_RULES_PATH_V2, DEFAULT_CANDIDATE_RULES_PATH_V2
        )

    def _events(self) -> pd.DataFrame:
        definitions = [
            # team, event, subevent, name, subevent name, period, seconds
            (1, 8, 85, "Pass", "Simple pass", "1H", 1.0),
            (2, 1, 12, "Duel", "Ground defending duel", "1H", 2.0),
            (2, 8, 85, "Pass", "Simple pass", "1H", 3.0),
            (1, 7, 71, "Others on the ball", "Clearance", "1H", 4.0),
            (2, 1, 13, "Duel", "Ground loose ball duel", "1H", 5.0),
            (1, 9, 91, "Save attempt", "Save attempt", "1H", 6.0),
            (1, 8, 85, "Pass", "Simple pass", "2H", 2701.0),
            (1, 3, 36, "Free Kick", "Throw in", "2H", 2702.0),
            (1, 8, 85, "Pass", "Simple pass", "2H", 2703.0),
        ]
        rows = []
        for index, values in enumerate(definitions):
            team, event_id, subevent_id, event_name, subevent_name, period, seconds = values
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
                    "start_x": 0.05 * index,
                    "start_y": 0.5,
                    "start_position_valid": True,
                    "end_x": 0.05 * index + 0.02,
                    "end_y": 0.5,
                    "end_position_valid": True,
                }
            )
        return pd.DataFrame(rows)

    def _tags_by_event(self) -> dict[int, set[int]]:
        return {
            100: {1801},
            101: {703},
            102: {1401, 1802},
            103: {1401, 1802},
            104: {701, 703},
            105: {1801},
            106: {1801},
            107: {1801},
            108: {1801},
        }

    def _tags(self) -> pd.DataFrame:
        rows = []
        for event_uid, tag_ids in self._tags_by_event().items():
            for order, tag_id in enumerate(sorted(tag_ids)):
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

    def test_contextual_candidates_and_conflicts(self) -> None:
        result = infer_match_possessions_v2(
            self._events(), self._tags_by_event(), (1, 2), self.rules
        )
        states = result.states.set_index("event_uid")

        # Pass interception + inaccurate both point to the opponent.
        self.assertEqual(states.loc[102, "candidate_status_after_event"], "single")
        self.assertEqual(int(states.loc[102, "candidate_team_after_event"]), 1)

        # Clearance interception and inaccurate point in opposite directions.
        self.assertEqual(
            states.loc[103, "candidate_status_after_event"], "conflicting"
        )
        self.assertTrue(pd.isna(states.loc[103, "candidate_team_after_event"]))

        # A duel containing both won and lost evidence is also unresolved.
        self.assertEqual(
            states.loc[104, "candidate_status_after_event"], "conflicting"
        )
        self.assertTrue(pd.isna(states.loc[104, "candidate_team_after_event"]))

        # Accurate save evidence nominates the goalkeeper team without switching.
        self.assertEqual(states.loc[105, "candidate_status_after_event"], "single")
        self.assertEqual(int(states.loc[105, "candidate_team_after_event"]), 1)
        self.assertFalse(bool(states.loc[105, "switch_confirmed"]))

    def test_period_reset_and_graph_ready_roles(self) -> None:
        result = infer_match_possessions_v2(
            self._events(), self._tags_by_event(), (1, 2), self.rules
        )
        states = result.states.set_index("event_uid")
        second_half = states.loc[106]
        self.assertEqual(second_half.control_state_before, "DEAD_BALL")
        self.assertTrue(pd.isna(second_half.owner_team_before_event))
        self.assertTrue(pd.isna(second_half.candidate_team_before_event))
        self.assertEqual(second_half.candidate_status_before_event, "none")

        clearance = states.loc[103]
        self.assertEqual(clearance.event_role, "contest")
        self.assertEqual(clearance.actor_relation_to_owner, "opponent")

    def test_v1_segmentation_and_transition_availability_are_unchanged(self) -> None:
        events = self._events()
        tags_by_event = self._tags_by_event()
        v1_states, v1_possessions = infer_match_possessions(
            events, tags_by_event, (1, 2), self.rules.base
        )
        result = infer_match_possessions_v2(
            events, tags_by_event, (1, 2), self.rules
        )
        report = validate_possessions_v2(
            events,
            self._tags(),
            pd.DataFrame(
                [{"match_id": 1, "team_id": 1}, {"match_id": 1, "team_id": 2}]
            ),
            result,
            self.rules,
            v1_states=v1_states,
            v1_possessions=v1_possessions,
            prefix_checks=5,
        )
        self.assertTrue(report["valid"], report["errors"])
        self.assertEqual(
            report["segmentation_identity"]["state_mismatch_count"], 0
        )
        self.assertEqual(
            report["segmentation_identity"]["possession_mismatch_count"], 0
        )

        starts = result.possessions.set_index("possession_uid")["start_event_index"]
        for transition in result.transitions.itertuples(index=False):
            self.assertEqual(
                int(transition.transition_event_index),
                int(starts.loc[transition.next_possession_uid]),
            )

    def test_model_safe_tables_exclude_audit_and_future_next_fields(self) -> None:
        result = infer_match_possessions_v2(
            self._events(), self._tags_by_event(), (1, 2), self.rules
        )
        for frame in (result.states, result.possessions):
            self.assertFalse(any("audit" in column for column in frame.columns))
            self.assertNotIn("next_possession_uid", frame.columns)
        self.assertIn("duration_seconds", result.audit.columns)
        self.assertIn("next_possession_uid", result.audit.columns)


if __name__ == "__main__":
    unittest.main()
