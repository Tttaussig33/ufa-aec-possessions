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
