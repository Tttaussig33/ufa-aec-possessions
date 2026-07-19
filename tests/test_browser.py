from __future__ import annotations

import pandas as pd

from ufa_aec_possessions.browser import create_team_aec_comparison_browser


def test_team_aec_comparison_browser_smoke():
    possessions = pd.DataFrame(
        {
            "team_id": ["glory", "glory"],
            "possession_id": ["p1", "p2"],
            "aec_per_throw": [0.3, 0.2],
            "total_aec": [0.6, 0.8],
        }
    )
    path = pd.DataFrame(
        {
            "possession_id": ["p1", "p1"],
            "possession_throw": [1, 2],
            "ThrowerX": [0, 5],
            "ThrowerY": [30, 60],
            "ReceiverX": [5, 0],
            "ReceiverY": [60, 105],
            "aec": [0.2, 0.4],
        }
    )
    comparison = {
        "by_metric": {
            "aec_per_throw": (possessions.iloc[[0]].copy(), {"glory": [path]}),
            "total_aec": (possessions.iloc[[1]].copy(), {"glory": [path]}),
        }
    }

    widget = create_team_aec_comparison_browser(comparison)

    assert len(widget.children) == 2


def test_team_aec_comparison_browser_ranks_dropdown_by_top_five_averages():
    left_possessions = pd.DataFrame(
        {
            "team_id": ["glory", "glory", "empire", "empire", "breeze", "breeze"],
            "possession_id": ["g1", "g2", "e1", "e2", "b1", "b2"],
            "aec_per_throw": [0.7, 0.6, 0.5, 0.4, 0.3, 0.2],
            "total_aec": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        }
    )
    right_possessions = pd.DataFrame(
        {
            "team_id": ["glory", "glory", "empire", "empire", "breeze", "breeze"],
            "possession_id": ["g3", "g4", "e3", "e4", "b3", "b4"],
            "aec_per_throw": [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
            "total_aec": [3.0, 2.8, 2.0, 1.8, 1.0, 0.8],
        }
    )
    path = pd.DataFrame(
        {
            "possession_id": ["g1", "g1"],
            "possession_throw": [1, 2],
            "ThrowerX": [0, 5],
            "ThrowerY": [30, 60],
            "ReceiverX": [5, 0],
            "ReceiverY": [60, 105],
            "aec": [0.2, 0.4],
        }
    )
    comparison = {
        "by_metric": {
            "aec_per_throw": (left_possessions, {team: [path] for team in ["glory", "empire", "breeze"]}),
            "total_aec": (right_possessions, {team: [path] for team in ["glory", "empire", "breeze"]}),
        }
    }

    widget = create_team_aec_comparison_browser(comparison)
    dropdown = widget.children[0].children[0]

    assert [label for label, _ in dropdown.options] == ["1. Glory", "2. Empire", "3. Breeze"]
