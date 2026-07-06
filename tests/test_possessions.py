from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ufa_aec_possessions import (
    add_possession_shape_features,
    build_possessions,
    filter_analysis_possessions,
    select_middle_aec_possessions,
    select_top_aec_possessions,
)


def synthetic_throws() -> pd.DataFrame:
    rows = []

    def add_throw(possession_num, throw_num, tx, ty, rx, ry, aec, o_line=True, receiver="r"):
        rows.append(
            {
                "GameID": "2026-01-01-GLY-TEST",
                "game_quarter": 1,
                "quarter_point": possession_num,
                "possession_num": possession_num,
                "possession_throw": throw_num,
                "is_home_team": True,
                "home_team_id": "glory",
                "away_team_id": "breeze",
                "Thrower": f"t{throw_num}",
                "Receiver": receiver,
                "ThrowerX": tx,
                "ThrowerY": ty,
                "ReceiverX": rx,
                "ReceiverY": ry,
                "x_diff": rx - tx,
                "y_diff": ry - ty,
                "throw_distance": float(np.hypot(rx - tx, ry - ty)),
                "aec": aec,
                "cp": 0.95,
                "o_line": o_line,
                "turnover": 0,
            }
        )

    # O-line, long-field, no huck scoring possession.
    add_throw(1, 1, 0, 30, 5, 60, 0.10)
    add_throw(1, 2, 5, 60, -5, 85, 0.20)
    add_throw(1, 3, -5, 85, 0, 105, 0.30)

    # O-line score that should be excluded by the huck filter.
    add_throw(2, 1, 0, 30, 0, 105, 0.80)

    # D-line score that should be excluded by the default line filter.
    add_throw(3, 1, 0, 30, 0, 55, 0.10, o_line=False)
    add_throw(3, 2, 0, 55, 0, 105, 0.30, o_line=False)
    return pd.DataFrame(rows)


def test_build_possessions_computes_metrics_and_default_filter():
    possessions, paths = build_possessions(synthetic_throws(), team_id="glory", outcomes=("goal",))

    assert len(possessions) == 3
    first = possessions.sort_values("possession_num").iloc[0]
    assert first["possession_id"] == "2026-01-01-GLY-TEST|1|1|1|True"
    assert first["line_type"] == "o_line"
    assert first["throw_count"] == 3
    assert first["total_aec"] == pytest.approx(0.60)
    assert first["aec_per_throw"] == pytest.approx(0.20)
    assert first["huck_count"] == 0
    assert first["reset_count"] == 0

    filtered, filtered_paths = filter_analysis_possessions(possessions, paths)
    assert filtered["possession_id"].tolist() == [first["possession_id"]]
    assert [path["possession_id"].iloc[0] for path in filtered_paths] == [first["possession_id"]]


def test_middle_selection_centers_ranked_window_and_aligns_paths():
    possessions = pd.DataFrame(
        {
            "possession_id": [f"p{i}" for i in range(7)],
            "aec_per_throw": [0.70, 0.10, 0.40, 0.20, 0.60, 0.30, 0.50],
        }
    )
    paths = [pd.DataFrame({"possession_id": [f"p{i}"], "possession_throw": [1]}) for i in range(7)]

    middle, middle_paths = select_middle_aec_possessions(possessions, paths, count=3)
    assert middle["possession_id"].tolist() == ["p5", "p2", "p6"]
    assert [path["possession_id"].iloc[0] for path in middle_paths] == ["p5", "p2", "p6"]

    top, _ = select_top_aec_possessions(possessions, paths, n=2)
    assert top["possession_id"].tolist() == ["p0", "p4"]


def test_shape_features_describe_path_geometry():
    possessions, paths = build_possessions(synthetic_throws(), team_id="glory", outcomes=("goal",))
    target = possessions[possessions["possession_num"].eq(1)].reset_index(drop=True)
    target_paths = [path for path in paths if path["possession_id"].iloc[0] == target["possession_id"].iloc[0]]

    enriched = add_possession_shape_features(target, target_paths)
    row = enriched.iloc[0]

    assert row["shape_width"] == pytest.approx(10.0)
    assert row["shape_middle_third_share"] == pytest.approx(1.0)
    assert row["shape_sideline_share"] == pytest.approx(0.0)
    assert row["shape_side_switches"] == 1
    assert row["shape_directness"] > 0.80
