from __future__ import annotations

import pandas as pd
import pytest

from ufa_aec_possessions import summarize_team_aec_per_throw


def test_summarize_team_aec_per_throw_ranks_weighted_team_rate():
    possessions = pd.DataFrame(
        {
            "possession_id": ["g1", "g2", "e1", "e2", "b1"],
            "team_id": ["glory", "glory", "empire", "empire", "breeze"],
            "GameID": ["game1", "game2", "game1", "game2", "game1"],
            "line_type": ["o_line", "o_line", "o_line", "d_line", "o_line"],
            "throw_count": [2, 8, 4, 4, 3],
            "total_aec": [1.0, 1.0, 1.2, 0.8, 0.3],
            "aec_per_throw": [0.5, 0.125, 0.3, 0.2, 0.1],
            "field_progress": [70, 65, 80, 50, 60],
            "huck_count": [0, 1, 0, 0, 0],
            "reset_count": [0, 2, 1, 1, 0],
        }
    )

    summary = summarize_team_aec_per_throw(possessions, min_possessions=2)

    assert summary["team_id"].tolist() == ["empire", "glory"]
    assert summary.loc[0, "team_aec_per_throw"] == pytest.approx(0.25)
    assert summary.loc[1, "team_aec_per_throw"] == pytest.approx(0.20)
    assert summary.loc[0, "possessions"] == 2
    assert summary.loc[0, "games"] == 2
    assert summary.loc[1, "huck_rate"] == pytest.approx(0.5)
