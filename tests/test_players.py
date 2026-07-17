import pandas as pd

from ufa_aec_possessions.players import (
    add_lag_contribution,
    summarize_lag_contribution_per_touch,
)


def _throws():
    return pd.DataFrame(
        [
            {
                "GameID": "g1",
                "game_quarter": 1,
                "quarter_point": 1,
                "possession_num": 1,
                "is_home_team": True,
                "possession_throw": 1,
                "Thrower": "alpha",
                "Receiver": "beta",
                "ReceiverY": 40,
                "home_team_id": "home",
                "away_team_id": "away",
                "aec": 0.10,
                "cp": 0.95,
                "cpoe": 0.02,
                "throw_distance": 12,
            },
            {
                "GameID": "g1",
                "game_quarter": 1,
                "quarter_point": 1,
                "possession_num": 1,
                "is_home_team": True,
                "possession_throw": 2,
                "Thrower": "beta",
                "Receiver": "gamma",
                "ReceiverY": 103,
                "home_team_id": "home",
                "away_team_id": "away",
                "aec": 0.40,
                "cp": 0.90,
                "cpoe": 0.03,
                "throw_distance": 24,
            },
            {
                "GameID": "g1",
                "game_quarter": 1,
                "quarter_point": 2,
                "possession_num": 1,
                "is_home_team": False,
                "possession_throw": 1,
                "Thrower": "alpha",
                "Receiver": "delta",
                "ReceiverY": 42,
                "home_team_id": "home",
                "away_team_id": "away",
                "aec": 0.20,
                "cp": 0.96,
                "cpoe": 0.01,
                "throw_distance": 11,
            },
        ]
    )


def test_add_lag_contribution_stays_within_possession():
    lagged = add_lag_contribution(_throws())

    assert lagged["lag_contribution"].tolist() == [0.40, 0.0, 0.0]
    assert lagged["has_next_throw"].tolist() == [True, False, False]
    assert lagged["sets_up_goal_throw"].tolist() == [True, False, False]
    assert lagged["team_id"].tolist() == ["home", "home", "away"]


def test_summarize_lag_contribution_per_touch():
    summary = summarize_lag_contribution_per_touch(_throws(), min_touches=1)

    alpha = summary[summary["thrower"].eq("alpha")].sort_values("team_id").reset_index(drop=True)
    assert alpha["touches"].tolist() == [1, 1]
    assert alpha["total_lag_contribution"].tolist() == [0.0, 0.40]
    assert alpha["lag_contribution_per_touch"].tolist() == [0.0, 0.40]

    beta = summary[summary["thrower"].eq("beta")].iloc[0]
    assert beta["touches"] == 1
    assert beta["total_lag_contribution"] == 0.0
