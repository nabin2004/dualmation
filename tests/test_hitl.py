"""
Unit tests for the HITL annotation tool.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from dualmation.training.hitl import HumanAnnotator


@pytest.fixture
def annotator(tmp_path):
    """Create a HumanAnnotator with a temporary file."""
    annot_file = tmp_path / "test_annotations.jsonl"
    return HumanAnnotator(annotation_file=annot_file)


def test_save_annotation(annotator):
    """Test saving a single annotation."""
    concept = "Test Concept"
    candidates = [{"code": "print(1)"}, {"code": "print(2)"}]
    preference = [1, 0] # 1 is better than 0
    
    annotator._save_annotation(concept, candidates, preference)
    
    assert annotator.annotation_file.exists()
    content = annotator.annotation_file.read_text().splitlines()
    assert len(content) == 1
    
    entry = json.loads(content[0])
    assert entry["concept"] == concept
    assert entry["best_idx"] == 1
    assert entry["ranking"] == preference


@patch("builtins.input", side_effect=["1 0"])
def test_interactive_session(mock_input, annotator):
    """Test a mock interactive session."""
    sessions = [{
        "concept": "Math",
        "candidates": [{"code": "x=1"}, {"code": "x=2"}]
    }]
    
    annotator.run_interactive(sessions)
    
    content = annotator.annotation_file.read_text().splitlines()
    assert len(content) == 1
    entry = json.loads(content[0])
    assert entry["ranking"] == [1, 0]
