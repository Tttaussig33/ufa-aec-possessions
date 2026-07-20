from __future__ import annotations

import pandas as pd

from ufa_aec_possessions.browser import create_team_aec_comparison_browser


def _path(possession_id: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "possession_id": [possession_id, possession_id],
            "possession_throw": [1, 2],
            "ThrowerX": [0, 5],
            "ThrowerY": [30, 60],
            "ReceiverX": [5, 0],
            "ReceiverY": [60, 105],
            "aec": [0.2, 0.4],
        }
    )


def _comparison_for_teams():
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
            "total_aec": [1.0, 0.8, 2.0, 1.8, 3.0, 2.8],
        }
    )
    paths_by_team = {team: [_path(f"{team}-path")] for team in ["glory", "empire", "breeze"]}
    return {
        "by_metric": {
            "aec_per_throw": (left_possessions, paths_by_team),
            "total_aec": (right_possessions, paths_by_team),
        }
    }


def test_team_aec_comparison_browser_smoke():
    widget = create_team_aec_comparison_browser(_comparison_for_teams())

    assert len(widget.children) == 3
    assert widget.get_title(0) == "Team metrics"
    assert widget.get_title(1) == "Compare aEC/T"
    assert widget.get_title(2) == "Compare total aEC"


def test_team_aec_comparison_browser_ranks_dropdown_by_top_five_averages():
    widget = create_team_aec_comparison_browser(_comparison_for_teams())
    metric_tab = widget.children[0]
    dropdown = metric_tab.children[0].children[0]
    compare_tab = widget.children[1]
    compare_left_dropdown = compare_tab.children[0].children[0]

    assert [label for label, _ in dropdown.options] == ["1. Glory", "2. Empire", "3. Breeze"]
    assert dropdown.options == compare_left_dropdown.options


def test_team_aec_comparison_browser_has_two_team_comparison_dropdowns():
    widget = create_team_aec_comparison_browser(_comparison_for_teams())
    compare_tab = widget.children[1]
    left_dropdown, right_dropdown = compare_tab.children[0].children

    assert left_dropdown.description == "Left"
    assert right_dropdown.description == "Right"
    assert left_dropdown.value == "glory"
    assert right_dropdown.value == "empire"
    assert [label for label, _ in left_dropdown.options] == ["1. Glory", "2. Empire", "3. Breeze"]
    assert [label for label, _ in right_dropdown.options] == ["1. Glory", "2. Empire", "3. Breeze"]


def test_team_aec_comparison_browser_total_aec_tab_ranks_by_total_aec_average():
    widget = create_team_aec_comparison_browser(_comparison_for_teams())
    compare_total_tab = widget.children[2]
    left_dropdown, right_dropdown = compare_total_tab.children[0].children

    assert left_dropdown.description == "Left"
    assert right_dropdown.description == "Right"
    assert left_dropdown.value == "breeze"
    assert right_dropdown.value == "empire"
    assert [label for label, _ in left_dropdown.options] == ["1. Breeze", "2. Empire", "3. Glory"]
    assert [label for label, _ in right_dropdown.options] == ["1. Breeze", "2. Empire", "3. Glory"]
