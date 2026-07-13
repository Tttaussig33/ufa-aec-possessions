from __future__ import annotations

import pandas as pd
import pytest

from ufa_aec_possessions import (
    summarize_team_aec_consistency,
    summarize_team_aec_per_throw,
    summarize_team_top_possessions,
)


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


def test_top_possessions_and_consistency_summaries_capture_different_questions():
    possessions = pd.DataFrame(
        {
            "possession_id": [f"spike{i}" for i in range(6)] + [f"steady{i}" for i in range(6)] + ["cheap1"],
            "team_id": ["spike"] * 6 + ["steady"] * 6 + ["spike"],
            "GameID": ["game1"] * 13,
            "line_type": ["o_line"] * 13,
            "throw_count": [5] * 12 + [1],
            "total_aec": [1.0, 0.95, 0.90, 0.10, 0.05, 0.0, 0.55, 0.54, 0.53, 0.52, 0.51, 0.50, 1.0],
            "aec_per_throw": [0.20, 0.19, 0.18, 0.02, 0.01, 0.0, 0.11, 0.108, 0.106, 0.104, 0.102, 0.10, 1.0],
            "huck_count": [0] * 13,
            "outcome": ["goal", "goal", "goal", "turnover", "turnover", "turnover"] * 2 + ["goal"],
        }
    )

    top_summary = summarize_team_top_possessions(
        possessions,
        top_n=3,
        min_possessions=3,
        min_throw_count=5,
    )
    consistency = summarize_team_aec_consistency(
        possessions,
        high_threshold=0.10,
        min_possessions=3,
        min_throw_count=5,
        top_n=3,
    )

    assert top_summary["team_id"].tolist() == ["spike", "steady"]
    assert top_summary.loc[0, "top_mean_metric"] == pytest.approx(0.19)
    assert consistency["team_id"].tolist() == ["steady", "spike"]
    assert consistency.loc[0, "median_metric"] == pytest.approx(0.105)
    assert consistency.loc[0, "high_metric_share"] == pytest.approx(1.0)
    assert consistency.loc[0, "goal_share"] == pytest.approx(0.5)
    assert consistency.loc[0, "turnover_share"] == pytest.approx(0.5)
