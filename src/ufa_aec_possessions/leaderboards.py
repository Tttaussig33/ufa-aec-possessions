from __future__ import annotations

import numpy as np
import pandas as pd


def _numeric(frame: pd.DataFrame, column: str, default=np.nan) -> pd.Series:
    if column in frame:
        return pd.to_numeric(frame[column], errors="coerce")
    return pd.Series(default, index=frame.index, dtype="float64")


def summarize_team_aec_per_throw(
    possessions: pd.DataFrame,
    *,
    min_possessions: int = 10,
    sort_by: str = "team_aec_per_throw",
) -> pd.DataFrame:
    """Summarize possession aEC/T by team from possession-level rows."""
    if possessions.empty:
        return pd.DataFrame()
    required = {"team_id", "possession_id", "total_aec", "throw_count"}
    missing = required - set(possessions.columns)
    if missing:
        raise ValueError(f"possessions is missing required columns: {sorted(missing)}")

    frame = possessions.copy()
    frame["total_aec"] = _numeric(frame, "total_aec", 0.0)
    frame["throw_count"] = _numeric(frame, "throw_count", 0.0)
    frame["aec_per_throw"] = _numeric(frame, "aec_per_throw")
    frame["field_progress"] = _numeric(frame, "field_progress")
    frame["huck_count"] = _numeric(frame, "huck_count", 0.0)
    frame["reset_count"] = _numeric(frame, "reset_count", 0.0)
    line_type = frame["line_type"] if "line_type" in frame else pd.Series("", index=frame.index)
    frame["is_o_line"] = line_type.astype(str).str.lower().eq("o_line")

    summary = (
        frame.groupby("team_id", dropna=False)
        .agg(
            possessions=("possession_id", "count"),
            games=("GameID", "nunique") if "GameID" in frame else ("possession_id", "count"),
            total_aec=("total_aec", "sum"),
            throws=("throw_count", "sum"),
            avg_throw_count=("throw_count", "mean"),
            avg_possession_aec_per_throw=("aec_per_throw", "mean"),
            median_possession_aec_per_throw=("aec_per_throw", "median"),
            avg_total_aec=("total_aec", "mean"),
            avg_field_progress=("field_progress", "mean"),
            huck_rate=("huck_count", lambda values: values.gt(0).mean()),
            resets_per_possession=("reset_count", "mean"),
            o_line_share=("is_o_line", "mean"),
        )
        .reset_index()
    )
    summary["team_aec_per_throw"] = summary["total_aec"] / summary["throws"].replace(0, np.nan)
    summary = summary[summary["possessions"].ge(min_possessions)].copy()
    if summary.empty:
        return summary
    if sort_by not in summary:
        raise ValueError(f"sort_by {sort_by!r} is not present in the team summary")
    summary = summary.sort_values(sort_by, ascending=False).reset_index(drop=True)
    summary["rank"] = range(1, len(summary) + 1)
    ordered_columns = [
        "rank",
        "team_id",
        "team_aec_per_throw",
        "avg_possession_aec_per_throw",
        "median_possession_aec_per_throw",
        "possessions",
        "games",
        "throws",
        "total_aec",
        "avg_total_aec",
        "avg_throw_count",
        "avg_field_progress",
        "huck_rate",
        "resets_per_possession",
        "o_line_share",
    ]
    return summary.reindex(columns=ordered_columns)
