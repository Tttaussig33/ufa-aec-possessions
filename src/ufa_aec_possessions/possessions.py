from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

FIELD_X_MIN = -26.65
FIELD_X_MAX = 26.65
FIELD_Y_MIN = 0
FIELD_Y_MAX = 120
ENDZONE_LOW_Y = 20
ENDZONE_HIGH_Y = 100


def _truthy_value(value) -> bool:
    if pd.isna(value):
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y", "o_line"}
    return bool(value)


def _has_text_value(value) -> bool:
    if pd.isna(value):
        return False
    return bool(str(value).strip())


def _numeric_column(frame: pd.DataFrame, column: str, default=np.nan) -> pd.Series:
    if column in frame:
        return pd.to_numeric(frame[column], errors="coerce")
    return pd.Series(default, index=frame.index, dtype="float64")


def _ensure_derived_throw_columns(throws: pd.DataFrame) -> pd.DataFrame:
    frame = throws.copy()
    if "GameID" not in frame and "game_id" in frame:
        frame["GameID"] = frame["game_id"]
    if "x_diff" not in frame and {"ThrowerX", "ReceiverX"}.issubset(frame):
        frame["x_diff"] = _numeric_column(frame, "ReceiverX") - _numeric_column(frame, "ThrowerX")
    if "y_diff" not in frame and {"ThrowerY", "ReceiverY"}.issubset(frame):
        frame["y_diff"] = _numeric_column(frame, "ReceiverY") - _numeric_column(frame, "ThrowerY")
    if "throw_distance" not in frame and {"x_diff", "y_diff"}.issubset(frame):
        frame["throw_distance"] = np.hypot(_numeric_column(frame, "x_diff"), _numeric_column(frame, "y_diff"))
    return frame


def _offense_team_id(frame: pd.DataFrame) -> str:
    home_team_id = frame["home_team_id"].iloc[0]
    away_team_id = frame["away_team_id"].iloc[0]
    is_home = _truthy_value(frame["is_home_team"].iloc[0])
    return str(home_team_id if is_home else away_team_id).lower()


def _possession_line_type(frame: pd.DataFrame) -> str:
    if "o_line" not in frame:
        return "unknown"
    values = frame["o_line"].dropna()
    if values.empty:
        return "unknown"
    return "o_line" if _truthy_value(values.iloc[0]) else "d_line"


def _throw_receiver_value(throw: pd.Series):
    for column in ["Receiver", "receiver", "receiver_id"]:
        if column in throw:
            return throw.get(column)
    return None


def _is_turnover_throw(throw: pd.Series) -> bool:
    for column in ["turnover", "Turnover"]:
        if column in throw:
            return _truthy_value(throw.get(column))
    return not _has_text_value(_throw_receiver_value(throw))


def _possession_outcome(path: pd.DataFrame) -> str:
    if path.empty:
        return "unknown"
    final_throw = path.iloc[-1]
    if _is_turnover_throw(final_throw):
        return "turnover"
    final_y = pd.to_numeric(pd.Series([final_throw.get("ReceiverY")]), errors="coerce").iloc[0]
    if pd.notna(final_y) and final_y > ENDZONE_HIGH_Y:
        return "goal"
    return "unknown"


def _fill_turnover_coordinates(throws: pd.DataFrame) -> pd.DataFrame:
    frame = throws.copy()
    for receiver_column, turnover_candidates in {
        "ReceiverX": ["TurnoverX", "turnoverX", "turnover_x"],
        "ReceiverY": ["TurnoverY", "turnoverY", "turnover_y"],
    }.items():
        if receiver_column not in frame:
            continue
        for turnover_column in turnover_candidates:
            if turnover_column in frame:
                frame[receiver_column] = frame[receiver_column].fillna(frame[turnover_column])
                break
    return frame


def _path_points(path: pd.DataFrame) -> pd.DataFrame:
    path = path.sort_values("possession_throw")
    if path.empty:
        return pd.DataFrame(columns=["x", "y", "aec", "cumulative_aec", "cp"])

    cumulative_aec = 0.0
    points = [
        {
            "x": path["ThrowerX"].iloc[0],
            "y": path["ThrowerY"].iloc[0],
            "aec": 0.0,
            "cumulative_aec": 0.0,
            "cp": np.nan,
        }
    ]
    for _, throw in path.iterrows():
        throw_aec = throw.get("aec", np.nan)
        if pd.notna(throw_aec):
            cumulative_aec += float(throw_aec)
        points.append(
            {
                "x": throw["ReceiverX"],
                "y": throw["ReceiverY"],
                "aec": throw_aec,
                "cumulative_aec": cumulative_aec,
                "cp": throw.get("cp", np.nan),
            }
        )
    return pd.DataFrame(points)


def _resample_path(points: pd.DataFrame, checkpoints: np.ndarray) -> pd.DataFrame:
    points = points.dropna(subset=["x", "y"]).copy()
    if points.empty:
        return pd.DataFrame()

    progress = points["y"].to_numpy(dtype=float)
    if progress[-1] == progress[0]:
        normalized = np.linspace(0, 1, len(points))
    else:
        normalized = (progress - progress[0]) / (progress[-1] - progress[0])
    normalized = np.maximum.accumulate(np.clip(normalized, 0, 1))

    dedup = pd.DataFrame(
        {
            "progress": normalized,
            "x": points["x"].to_numpy(dtype=float),
            "y": points["y"].to_numpy(dtype=float),
            "cumulative_aec": points["cumulative_aec"].to_numpy(dtype=float),
            "cp": points["cp"].to_numpy(dtype=float),
        }
    ).drop_duplicates("progress", keep="last")

    if len(dedup) == 1:
        x_values = np.repeat(dedup["x"].iloc[0], len(checkpoints))
        y_values = np.repeat(dedup["y"].iloc[0], len(checkpoints))
    else:
        x_values = np.interp(checkpoints, dedup["progress"], dedup["x"])
        y_values = np.interp(checkpoints, dedup["progress"], dedup["y"])

    return pd.DataFrame(
        {
            "checkpoint": checkpoints,
            "x": x_values,
            "y": y_values,
            "cumulative_aec": np.interp(checkpoints, dedup["progress"], dedup["cumulative_aec"]),
            "cp": np.interp(checkpoints, dedup["progress"], dedup["cp"].ffill().bfill()),
        }
    )


def build_possessions(
    throws: pd.DataFrame,
    team_id: str | None = None,
    outcomes: Iterable[str] | None = None,
) -> tuple[pd.DataFrame, list[pd.DataFrame]]:
    """Build possession-level rows and real possession paths from throw rows."""
    if throws.empty:
        return pd.DataFrame(), []

    outcome_filter = {str(outcome).lower() for outcome in outcomes} if outcomes is not None else None
    frame = _ensure_derived_throw_columns(_fill_turnover_coordinates(throws))
    required = [
        "GameID",
        "game_quarter",
        "quarter_point",
        "possession_num",
        "is_home_team",
        "ThrowerX",
        "ThrowerY",
        "ReceiverX",
        "ReceiverY",
        "possession_throw",
        "home_team_id",
        "away_team_id",
    ]
    missing = [column for column in required if column not in frame]
    if missing:
        raise ValueError(f"throws is missing required columns: {missing}")

    frame = frame.dropna(subset=["ThrowerX", "ThrowerY", "ReceiverX", "ReceiverY", "possession_throw"])
    group_columns = ["GameID", "game_quarter", "quarter_point", "possession_num", "is_home_team"]

    possession_rows: list[dict] = []
    paths: list[pd.DataFrame] = []
    for key, group in frame.groupby(group_columns, dropna=False):
        path = group.sort_values("possession_throw").copy()
        outcome = _possession_outcome(path)
        if outcome_filter is not None and outcome not in outcome_filter:
            continue

        offense_team = _offense_team_id(path)
        if team_id is not None and offense_team != team_id.lower():
            continue

        line_type = _possession_line_type(path)
        aec = _numeric_column(path, "aec", 0.0)
        cp = _numeric_column(path, "cp")
        x_diff = _numeric_column(path, "x_diff")
        y_diff = _numeric_column(path, "y_diff")
        throw_distance = _numeric_column(path, "throw_distance")
        throw_count = len(path)
        total_aec = aec.sum()
        start_x = _numeric_column(path, "ThrowerX").iloc[0]
        start_y = _numeric_column(path, "ThrowerY").iloc[0]
        end_x = _numeric_column(path, "ReceiverX").iloc[-1]
        end_y = _numeric_column(path, "ReceiverY").iloc[-1]
        possession_id = "|".join(str(value) for value in key)

        possession_rows.append(
            {
                "possession_id": possession_id,
                "GameID": key[0],
                "team_id": offense_team,
                "start_timestamp": path["start_timestamp"].iloc[0] if "start_timestamp" in path else None,
                "game_quarter": key[1],
                "quarter_point": key[2],
                "possession_num": key[3],
                "is_home_team": key[4],
                "line_type": line_type,
                "outcome": outcome,
                "is_goal": outcome == "goal",
                "is_turnover": outcome == "turnover",
                "start_x": start_x,
                "start_y": start_y,
                "end_x": end_x,
                "end_y": end_y,
                "field_progress": end_y - start_y,
                "throw_count": throw_count,
                "total_aec": total_aec,
                "aec_per_throw": total_aec / throw_count if throw_count else np.nan,
                "mean_cp": cp.mean(),
                "risk_adjusted_aec_per_throw": total_aec / throw_count * cp.mean() if throw_count else np.nan,
                "total_yards": y_diff.sum(),
                "yards_per_throw": y_diff.mean(),
                "total_throw_distance": throw_distance.sum(),
                "avg_throw_distance": throw_distance.mean(),
                "max_throw_distance": throw_distance.max(),
                "huck_count": int(throw_distance.ge(40).sum()),
                "reset_count": int(y_diff.lt(0).sum()),
                "lateral_yards": x_diff.abs().sum(),
            }
        )
        path["possession_id"] = possession_id
        path["team_id"] = offense_team
        path["line_type"] = line_type
        path["outcome"] = outcome
        paths.append(path)

    return pd.DataFrame(possession_rows), paths


def build_scoring_possessions(throws: pd.DataFrame, team_id: str | None = None) -> tuple[pd.DataFrame, list[pd.DataFrame]]:
    """Build only goal-scoring possessions."""
    return build_possessions(throws, team_id=team_id, outcomes=("goal",))


def _path_shape_feature_row(path: pd.DataFrame, checkpoints: np.ndarray) -> dict:
    path = path.sort_values("possession_throw").copy()
    points = _path_points(path).dropna(subset=["x", "y"])
    if points.empty:
        return {}

    sampled = _resample_path(points, checkpoints)
    if sampled.empty:
        return {}

    x_values = points["x"].to_numpy(dtype=float)
    y_values = points["y"].to_numpy(dtype=float)
    thrower_x = _numeric_column(path, "ThrowerX")
    receiver_x = _numeric_column(path, "ReceiverX")
    receiver_y = _numeric_column(path, "ReceiverY")
    x_diff = _numeric_column(path, "x_diff")
    y_diff = _numeric_column(path, "y_diff")
    throw_distance = _numeric_column(path, "throw_distance")

    total_distance = throw_distance.sum()
    net_distance = float(np.hypot(x_values[-1] - x_values[0], y_values[-1] - y_values[0]))
    directness = net_distance / total_distance if total_distance else np.nan
    field_progress = y_values[-1] - y_values[0]
    lateral_yards = x_diff.abs().sum()

    point_x = pd.Series(np.r_[thrower_x.to_numpy(), receiver_x.to_numpy()])
    point_y = pd.Series(np.r_[path["ThrowerY"].to_numpy(), receiver_y.to_numpy()])
    valid_points = pd.DataFrame({"x": point_x, "y": point_y}).dropna()
    middle_third_share = valid_points["x"].abs().le(8.88).mean()
    sideline_share = valid_points["x"].abs().ge(17.77).mean()
    left_side_share = valid_points["x"].lt(-8.88).mean()
    right_side_share = valid_points["x"].gt(8.88).mean()

    signs = np.sign(x_values)
    non_zero_signs = signs[signs != 0]
    side_switches = np.count_nonzero(non_zero_signs[1:] != non_zero_signs[:-1]) if len(non_zero_signs) > 1 else 0

    red_zone_rows = path[receiver_y.ge(ENDZONE_HIGH_Y - 20)]
    red_zone_entry_x = _numeric_column(red_zone_rows, "ReceiverX").iloc[0] if not red_zone_rows.empty else np.nan

    features = {
        "shape_start_x": x_values[0],
        "shape_start_y": y_values[0],
        "shape_end_x": x_values[-1],
        "shape_end_y": y_values[-1],
        "shape_width": np.nanmax(x_values) - np.nanmin(x_values),
        "shape_directness": directness,
        "shape_lateral_per_yard": lateral_yards / abs(field_progress) if field_progress else np.nan,
        "shape_side_switches": side_switches,
        "shape_middle_third_share": middle_third_share,
        "shape_sideline_share": sideline_share,
        "shape_left_side_share": left_side_share,
        "shape_right_side_share": right_side_share,
        "shape_red_zone_entry_x": red_zone_entry_x,
        "shape_red_zone_throws": int(receiver_y.ge(ENDZONE_HIGH_Y - 20).sum()),
        "shape_backwards_share": y_diff.lt(0).mean(),
        "shape_large_gain_share": throw_distance.ge(25).mean(),
    }
    for _, point in sampled.iterrows():
        label = int(round(point["checkpoint"] * 100))
        features[f"shape_x_{label:03d}"] = point["x"]
        features[f"shape_y_{label:03d}"] = point["y"]
    return features


def calculate_possession_shape_features(
    possessions: pd.DataFrame,
    paths: list[pd.DataFrame],
    checkpoints: Iterable[float] | None = None,
) -> pd.DataFrame:
    """Return one row of geometry-first features per possession."""
    if possessions.empty:
        return pd.DataFrame()

    checkpoint_values = np.asarray(checkpoints if checkpoints is not None else np.linspace(0, 1, 8), dtype=float)
    rows = []
    for path in paths:
        if path.empty or "possession_id" not in path:
            continue
        feature_row = _path_shape_feature_row(path, checkpoint_values)
        if feature_row:
            feature_row["possession_id"] = path["possession_id"].iloc[0]
            rows.append(feature_row)

    if not rows:
        return pd.DataFrame(index=possessions.index)

    shape_features = pd.DataFrame(rows).set_index("possession_id")
    aligned = possessions[["possession_id"]].join(shape_features, on="possession_id")
    return aligned.drop(columns=["possession_id"]).apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)


def add_possession_shape_features(
    possessions: pd.DataFrame,
    paths: list[pd.DataFrame],
    checkpoints: Iterable[float] | None = None,
) -> pd.DataFrame:
    """Attach geometry-first possession features to a possession table."""
    if possessions.empty:
        return possessions.copy()
    enriched = possessions.copy()
    shape_features = calculate_possession_shape_features(enriched, paths, checkpoints=checkpoints)
    for column in shape_features:
        enriched[column] = shape_features[column].to_numpy()
    return enriched
