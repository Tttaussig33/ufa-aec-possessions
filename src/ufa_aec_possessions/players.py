from __future__ import annotations

import numpy as np
import pandas as pd

from ufa_aec_possessions.possessions import _ensure_derived_throw_columns, _truthy_value

DEFAULT_POSSESSION_COLUMNS = [
    "GameID",
    "game_quarter",
    "quarter_point",
    "possession_num",
    "is_home_team",
]


def _numeric(frame: pd.DataFrame, column: str, default=np.nan) -> pd.Series:
    if column in frame:
        return pd.to_numeric(frame[column], errors="coerce")
    return pd.Series(default, index=frame.index, dtype="float64")


def _offense_team_series(frame: pd.DataFrame) -> pd.Series:
    if {"home_team_id", "away_team_id", "is_home_team"}.issubset(frame.columns):
        is_home = frame["is_home_team"].map(_truthy_value)
        return pd.Series(
            np.where(is_home, frame["home_team_id"], frame["away_team_id"]),
            index=frame.index,
        ).astype(str).str.lower()
    if "team_id" in frame:
        return frame["team_id"].astype(str).str.lower()
    return pd.Series("", index=frame.index, dtype="object")


def add_lag_contribution(
    throws: pd.DataFrame,
    *,
    possession_columns: list[str] | None = None,
    aec_column: str = "aec",
) -> pd.DataFrame:
    """Add one-throw lag contribution fields to Shown Space throw rows.

    Lag contribution credits a thrower with the aEC of the next throw in the
    same possession. This is a transparent approximation of the "sets up a
    teammate" idea: a swing, reset, or continuation pass gets credit when the
    receiver turns it into value on the following throw.
    """
    if throws.empty:
        return throws.copy()

    possession_columns = possession_columns or DEFAULT_POSSESSION_COLUMNS
    missing = [column for column in possession_columns if column not in throws]
    if missing:
        raise ValueError(f"throws is missing possession columns: {missing}")
    if "possession_throw" not in throws:
        raise ValueError("throws is missing required column: 'possession_throw'")
    if "Thrower" not in throws:
        raise ValueError("throws is missing required column: 'Thrower'")
    if aec_column not in throws:
        raise ValueError(f"throws is missing required aEC column: {aec_column!r}")

    frame = _ensure_derived_throw_columns(throws).copy()
    frame["_original_order"] = np.arange(len(frame))
    frame[aec_column] = _numeric(frame, aec_column, 0.0).fillna(0.0)
    frame["team_id"] = _offense_team_series(frame)
    frame = frame.sort_values(possession_columns + ["possession_throw", "_original_order"]).copy()

    grouped = frame.groupby(possession_columns, dropna=False)
    frame["next_throw_aec"] = grouped[aec_column].shift(-1)
    frame["next_thrower"] = grouped["Thrower"].shift(-1)
    if "Receiver" in frame:
        frame["next_receiver"] = grouped["Receiver"].shift(-1)
    else:
        frame["next_receiver"] = pd.NA
    if "ReceiverY" in frame:
        frame["next_receiver_y"] = grouped["ReceiverY"].shift(-1)
    else:
        frame["next_receiver_y"] = np.nan

    frame["has_next_throw"] = frame["next_throw_aec"].notna()
    frame["lag_contribution"] = frame["next_throw_aec"].fillna(0.0)
    frame["sets_up_goal_throw"] = (
        _numeric(frame, "next_receiver_y").gt(100)
        & frame["next_receiver"].notna()
        & frame["next_receiver"].astype(str).str.strip().ne("")
    )

    return frame.sort_values("_original_order").drop(columns=["_original_order"]).reset_index(drop=True)


def summarize_lag_contribution_per_touch(
    throws: pd.DataFrame,
    *,
    min_touches: int = 100,
    sort_by: str = "lag_contribution_per_touch",
) -> pd.DataFrame:
    """Rank throwers by lag contribution per touch.

    In this summary, a touch is a throw attempt by the player. The lag
    contribution attached to that touch is the aEC of the next throw in the
    same possession.
    """
    if throws.empty:
        return pd.DataFrame()

    frame = add_lag_contribution(throws)
    frame["thrower"] = frame["Thrower"].astype(str)
    frame = frame[frame["thrower"].str.strip().ne("")].copy()
    frame["aec"] = _numeric(frame, "aec", 0.0).fillna(0.0)
    frame["cp"] = _numeric(frame, "cp")
    frame["cpoe"] = _numeric(frame, "cpoe")
    frame["throw_distance"] = _numeric(frame, "throw_distance")
    frame["is_completion"] = frame.get("Receiver", pd.Series("", index=frame.index)).astype(str).str.strip().ne("")
    frame["is_huck"] = frame["throw_distance"].ge(40)

    summary = (
        frame.groupby(["team_id", "thrower"], dropna=False)
        .agg(
            touches=("thrower", "count"),
            total_lag_contribution=("lag_contribution", "sum"),
            setup_opportunities=("has_next_throw", "sum"),
            goal_setups=("sets_up_goal_throw", "sum"),
            total_thrower_aec=("aec", "sum"),
            completions=("is_completion", "sum"),
            avg_cp=("cp", "mean"),
            avg_cpoe=("cpoe", "mean"),
            huck_rate=("is_huck", "mean"),
            avg_throw_distance=("throw_distance", "mean"),
        )
        .reset_index()
    )
    summary["lag_contribution_per_touch"] = (
        summary["total_lag_contribution"] / summary["touches"].replace(0, np.nan)
    )
    summary["lag_contribution_per_100_touches"] = summary["lag_contribution_per_touch"] * 100
    summary["thrower_aec_per_touch"] = summary["total_thrower_aec"] / summary["touches"].replace(0, np.nan)
    summary["completion_rate"] = summary["completions"] / summary["touches"].replace(0, np.nan)
    summary["goal_setup_rate"] = summary["goal_setups"] / summary["touches"].replace(0, np.nan)
    summary = summary[summary["touches"].ge(min_touches)].copy()
    if summary.empty:
        return summary
    if sort_by not in summary:
        raise ValueError(f"sort_by {sort_by!r} is not present in the lag contribution summary")
    summary = summary.sort_values(sort_by, ascending=False).reset_index(drop=True)
    summary["rank"] = range(1, len(summary) + 1)

    ordered_columns = [
        "rank",
        "team_id",
        "thrower",
        "lag_contribution_per_touch",
        "lag_contribution_per_100_touches",
        "total_lag_contribution",
        "touches",
        "setup_opportunities",
        "goal_setups",
        "goal_setup_rate",
        "thrower_aec_per_touch",
        "total_thrower_aec",
        "completion_rate",
        "avg_cp",
        "avg_cpoe",
        "huck_rate",
        "avg_throw_distance",
    ]
    return summary.reindex(columns=ordered_columns)
