from __future__ import annotations

import pandas as pd

from ufa_aec_possessions.possessions import add_possession_shape_features

DEFAULT_METRIC = "aec_per_throw"


def _path_lookup(paths: list[pd.DataFrame]) -> dict[str, pd.DataFrame]:
    return {str(path["possession_id"].iloc[0]): path for path in paths if not path.empty and "possession_id" in path}


def _aligned_paths(possessions: pd.DataFrame, paths: list[pd.DataFrame]) -> list[pd.DataFrame]:
    if possessions.empty or "possession_id" not in possessions:
        return []
    lookup = _path_lookup(paths)
    return [lookup[possession_id] for possession_id in possessions["possession_id"].astype(str) if possession_id in lookup]


def filter_analysis_possessions(
    possessions: pd.DataFrame,
    paths: list[pd.DataFrame] | None = None,
    *,
    team_id: str | None = None,
    outcomes: tuple[str, ...] | None = ("goal",),
    line_type: str | None = "o_line",
    long_field_only: bool = True,
    max_start_y: float = 45,
    min_field_progress: float = 50,
    exclude_hucks: bool = True,
) -> tuple[pd.DataFrame, list[pd.DataFrame]]:
    """Apply the default high-quality AEC possession analysis filter."""
    filtered = possessions.copy()
    if filtered.empty:
        return filtered, []

    if team_id is not None and "team_id" in filtered:
        filtered = filtered[filtered["team_id"].astype(str).str.lower().eq(team_id.lower())]
    if outcomes is not None and "outcome" in filtered:
        allowed = {outcome.lower() for outcome in outcomes}
        filtered = filtered[filtered["outcome"].astype(str).str.lower().isin(allowed)]
    if line_type is not None and "line_type" in filtered:
        filtered = filtered[filtered["line_type"].astype(str).str.lower().eq(line_type.lower())]
    if long_field_only:
        filtered = filtered[
            pd.to_numeric(filtered["start_y"], errors="coerce").le(max_start_y)
            & pd.to_numeric(filtered["field_progress"], errors="coerce").ge(min_field_progress)
        ]
    if exclude_hucks and "huck_count" in filtered:
        filtered = filtered[pd.to_numeric(filtered["huck_count"], errors="coerce").fillna(0).eq(0)]

    filtered = filtered.reset_index(drop=True)
    return filtered, _aligned_paths(filtered, paths or [])


def _rank_possessions(possessions: pd.DataFrame, metric: str, ascending: bool) -> pd.DataFrame:
    if possessions.empty:
        return possessions.copy()
    if metric not in possessions:
        raise ValueError(f"metric {metric!r} is not present in possessions")
    ranked = possessions.copy()
    ranked[metric] = pd.to_numeric(ranked[metric], errors="coerce")
    return ranked.sort_values(metric, ascending=ascending, na_position="last").reset_index(drop=True)


def select_top_aec_possessions(
    possessions: pd.DataFrame,
    paths: list[pd.DataFrame],
    *,
    metric: str = DEFAULT_METRIC,
    n: int = 5,
    ascending: bool = False,
) -> tuple[pd.DataFrame, list[pd.DataFrame]]:
    """Select the top ranked possessions and return rows plus aligned paths."""
    selected = _rank_possessions(possessions, metric, ascending).head(n).reset_index(drop=True)
    return selected, _aligned_paths(selected, paths)


def select_middle_aec_possessions(
    possessions: pd.DataFrame,
    paths: list[pd.DataFrame],
    *,
    metric: str = DEFAULT_METRIC,
    count: int = 5,
) -> tuple[pd.DataFrame, list[pd.DataFrame]]:
    """Select the centered window after sorting possessions by AEC metric."""
    ranked = _rank_possessions(possessions, metric, ascending=True)
    middle_count = min(count, len(ranked))
    start = max((len(ranked) - middle_count) // 2, 0)
    selected = ranked.iloc[start : start + middle_count].reset_index(drop=True)
    return selected, _aligned_paths(selected, paths)


def build_aec_possession_sets(
    possessions: pd.DataFrame,
    paths: list[pd.DataFrame],
    *,
    top_n: int = 5,
    middle_count: int = 5,
    metric: str = DEFAULT_METRIC,
    add_shape_features: bool = True,
    **filter_kwargs,
) -> dict[str, tuple[pd.DataFrame, list[pd.DataFrame]]]:
    """Return filtered, highest, and middle possession sets for one analysis run."""
    filtered, filtered_paths = filter_analysis_possessions(possessions, paths, **filter_kwargs)
    if add_shape_features and not filtered.empty:
        filtered = add_possession_shape_features(filtered, filtered_paths)
    highest = select_top_aec_possessions(filtered, filtered_paths, metric=metric, n=top_n)
    middle = select_middle_aec_possessions(filtered, filtered_paths, metric=metric, count=middle_count)
    return {"filtered": (filtered, filtered_paths), "highest": highest, "middle": middle}


def select_top_aec_possessions_by_team(
    possessions: pd.DataFrame,
    paths: list[pd.DataFrame],
    *,
    metric: str = DEFAULT_METRIC,
    n: int = 5,
    add_shape_features: bool = True,
    **filter_kwargs,
) -> tuple[pd.DataFrame, dict[str, list[pd.DataFrame]]]:
    """Select each team's top ranked possessions from one league-wide pool."""
    filtered, filtered_paths = filter_analysis_possessions(
        possessions,
        paths,
        team_id=None,
        **filter_kwargs,
    )
    if filtered.empty:
        return filtered, {}

    if add_shape_features:
        filtered = add_possession_shape_features(filtered, filtered_paths)

    team_rows = []
    paths_by_team: dict[str, list[pd.DataFrame]] = {}
    for team_id, team_possessions in filtered.groupby("team_id", dropna=False):
        team_label = str(team_id)
        selected, selected_paths = select_top_aec_possessions(
            team_possessions,
            filtered_paths,
            metric=metric,
            n=n,
        )
        selected = selected.copy()
        selected["team_rank"] = range(1, len(selected) + 1)
        team_rows.append(selected)
        paths_by_team[team_label] = selected_paths

    if not team_rows:
        return pd.DataFrame(), {}

    league_top = pd.concat(team_rows, ignore_index=True)
    sort_columns = ["team_id", "team_rank"]
    return league_top.sort_values(sort_columns).reset_index(drop=True), paths_by_team

