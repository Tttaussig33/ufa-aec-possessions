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


def summarize_team_top_possessions(
    possessions: pd.DataFrame,
    *,
    metric: str = "aec_per_throw",
    top_n: int = 5,
    min_possessions: int = 5,
    min_throw_count: int | None = None,
) -> pd.DataFrame:
    """Summarize each team's top-N possession values for one metric."""
    if possessions.empty:
        return pd.DataFrame()
    required = {"team_id", "possession_id", metric}
    missing = required - set(possessions.columns)
    if missing:
        raise ValueError(f"possessions is missing required columns: {sorted(missing)}")

    frame = possessions.copy()
    frame[metric] = _numeric(frame, metric)
    frame["total_aec"] = _numeric(frame, "total_aec", 0.0)
    frame["throw_count"] = _numeric(frame, "throw_count", 0.0)
    frame["huck_count"] = _numeric(frame, "huck_count", 0.0)
    frame = frame.dropna(subset=[metric])
    if min_throw_count is not None:
        frame = frame[frame["throw_count"].ge(min_throw_count)].copy()

    eligible_counts = frame.groupby("team_id", dropna=False)["possession_id"].count()
    eligible_teams = eligible_counts[eligible_counts.ge(min_possessions)].index
    frame = frame[frame["team_id"].isin(eligible_teams)].copy()
    if frame.empty:
        return pd.DataFrame()

    ranked = frame.sort_values(["team_id", metric], ascending=[True, False]).copy()
    ranked["team_metric_rank"] = ranked.groupby("team_id", dropna=False).cumcount() + 1
    top = ranked[ranked["team_metric_rank"].le(top_n)].copy()

    summary = (
        top.groupby("team_id", dropna=False)
        .agg(
            top_possessions=("possession_id", "count"),
            top_mean_metric=(metric, "mean"),
            top_median_metric=(metric, "median"),
            top_floor_metric=(metric, "min"),
            top_ceiling_metric=(metric, "max"),
            top_total_aec=("total_aec", "sum"),
            top_throws=("throw_count", "sum"),
            top_avg_throw_count=("throw_count", "mean"),
        )
        .reset_index()
    )

    all_summary = summarize_team_aec_per_throw(frame, min_possessions=1)
    all_columns = [
        "team_id",
        "team_aec_per_throw",
        "avg_possession_aec_per_throw",
        "median_possession_aec_per_throw",
        "possessions",
        "games",
        "throws",
        "huck_rate",
        "o_line_share",
    ]
    summary = summary.merge(all_summary.reindex(columns=all_columns), on="team_id", how="left")
    summary = summary.sort_values("top_mean_metric", ascending=False).reset_index(drop=True)
    summary["rank"] = range(1, len(summary) + 1)
    ordered_columns = [
        "rank",
        "team_id",
        "top_possessions",
        "top_mean_metric",
        "top_median_metric",
        "top_floor_metric",
        "top_ceiling_metric",
        "top_total_aec",
        "top_throws",
        "top_avg_throw_count",
        "team_aec_per_throw",
        "avg_possession_aec_per_throw",
        "median_possession_aec_per_throw",
        "possessions",
        "games",
        "throws",
        "huck_rate",
        "o_line_share",
    ]
    return summary.reindex(columns=ordered_columns)


def summarize_team_aec_consistency(
    possessions: pd.DataFrame,
    *,
    metric: str = "aec_per_throw",
    high_threshold: float | None = None,
    min_possessions: int = 20,
    min_throw_count: int | None = None,
    top_n: int = 10,
    sort_by: str = "median_metric",
) -> pd.DataFrame:
    """Summarize how consistently each team produces high possession aEC/T."""
    if possessions.empty:
        return pd.DataFrame()
    required = {"team_id", "possession_id", metric, "total_aec", "throw_count"}
    missing = required - set(possessions.columns)
    if missing:
        raise ValueError(f"possessions is missing required columns: {sorted(missing)}")

    frame = possessions.copy()
    frame[metric] = _numeric(frame, metric)
    frame["total_aec"] = _numeric(frame, "total_aec", 0.0)
    frame["throw_count"] = _numeric(frame, "throw_count", 0.0)
    frame["huck_count"] = _numeric(frame, "huck_count", 0.0)
    line_type = frame["line_type"] if "line_type" in frame else pd.Series("", index=frame.index)
    frame["is_o_line"] = line_type.astype(str).str.lower().eq("o_line")
    outcome = frame["outcome"] if "outcome" in frame else pd.Series("", index=frame.index)
    frame["is_goal"] = outcome.astype(str).str.lower().eq("goal")
    frame["is_turnover"] = outcome.astype(str).str.lower().eq("turnover")
    frame = frame.dropna(subset=[metric])
    if min_throw_count is not None:
        frame = frame[frame["throw_count"].ge(min_throw_count)].copy()
    if frame.empty:
        return pd.DataFrame()

    if high_threshold is None:
        high_threshold = float(frame[metric].quantile(0.75))

    def top_mean(values: pd.Series) -> float:
        return values.sort_values(ascending=False).head(top_n).mean()

    summary = (
        frame.groupby("team_id", dropna=False)
        .agg(
            possessions=("possession_id", "count"),
            games=("GameID", "nunique") if "GameID" in frame else ("possession_id", "count"),
            throws=("throw_count", "sum"),
            total_aec=("total_aec", "sum"),
            team_aec_per_throw=("total_aec", lambda values: values.sum()),
            avg_metric=(metric, "mean"),
            median_metric=(metric, "median"),
            p75_metric=(metric, lambda values: values.quantile(0.75)),
            p90_metric=(metric, lambda values: values.quantile(0.90)),
            top_n_mean_metric=(metric, top_mean),
            high_metric_share=(metric, lambda values: values.ge(high_threshold).mean()),
            avg_throw_count=("throw_count", "mean"),
            huck_rate=("huck_count", lambda values: values.gt(0).mean()),
            o_line_share=("is_o_line", "mean"),
            goal_share=("is_goal", "mean"),
            turnover_share=("is_turnover", "mean"),
        )
        .reset_index()
    )
    summary["team_aec_per_throw"] = summary["total_aec"] / summary["throws"].replace(0, np.nan)
    summary["high_metric_threshold"] = high_threshold
    summary = summary[summary["possessions"].ge(min_possessions)].copy()
    if summary.empty:
        return summary
    if sort_by not in summary:
        raise ValueError(f"sort_by {sort_by!r} is not present in the consistency summary")
    summary = summary.sort_values(sort_by, ascending=False).reset_index(drop=True)
    summary["rank"] = range(1, len(summary) + 1)
    ordered_columns = [
        "rank",
        "team_id",
        "median_metric",
        "p75_metric",
        "p90_metric",
        "top_n_mean_metric",
        "high_metric_share",
        "team_aec_per_throw",
        "avg_metric",
        "possessions",
        "games",
        "throws",
        "total_aec",
        "avg_throw_count",
        "huck_rate",
        "o_line_share",
        "goal_share",
        "turnover_share",
        "high_metric_threshold",
    ]
    return summary.reindex(columns=ordered_columns)
