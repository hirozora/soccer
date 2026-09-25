from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from football_benchmark.constants import DEFAULT_ARTIFACT_PATH, FEASIBILITY_ARTIFACT_PATH
from football_benchmark.protocol import ProtocolArtifacts


@pytest.fixture(scope="session")
def artifacts() -> ProtocolArtifacts:
    if not DEFAULT_ARTIFACT_PATH.exists():
        pytest.skip("protocol artifact has not been built")
    return ProtocolArtifacts.load(DEFAULT_ARTIFACT_PATH)


@pytest.fixture(scope="session")
def feasibility_artifacts() -> ProtocolArtifacts:
    if not FEASIBILITY_ARTIFACT_PATH.exists():
        pytest.skip("feasibility protocol artifact has not been built")
    return ProtocolArtifacts.load(FEASIBILITY_ARTIFACT_PATH)
