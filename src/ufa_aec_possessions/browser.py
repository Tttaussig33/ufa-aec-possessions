from __future__ import annotations

import pandas as pd

from ufa_aec_possessions.plotting import plot_side_by_side_paths


def _label_metric_paths(
    possessions: pd.DataFrame,
    paths: list[pd.DataFrame],
    metric: str,
    prefix: str,
) -> dict[str, pd.DataFrame]:
    lookup = {path["possession_id"].iloc[0]: path for path in paths if not path.empty}
    labeled = {}
    for rank, (_, row) in enumerate(possessions.reset_index(drop=True).iterrows(), start=1):
        possession_id = row["possession_id"]
        if possession_id in lookup:
            labeled[f"{prefix} {rank}: {row[metric]:.3f}"] = lookup[possession_id]
    return labeled


def _metric_average(possessions: pd.DataFrame, metric: str) -> float:
    if possessions.empty or metric not in possessions:
        return float("nan")
    return pd.to_numeric(possessions[metric], errors="coerce").mean()


def _format_metric_average(value: float, metric: str, window_label: str = "Top 5") -> str:
    label = "aEC/T" if metric == "aec_per_throw" else metric.replace("_", " ")
    if pd.isna(value):
        return f"{window_label} average {label}: -"
    return f"{window_label} average {label}: {value:.3f}"


def _team_top_average_summary(
    team_ids: list[str],
    left_possessions: pd.DataFrame,
    right_possessions: pd.DataFrame,
    left_metric: str,
    right_metric: str,
) -> pd.DataFrame:
    rows = []
    left_team_ids = left_possessions.get("team_id", pd.Series(dtype=str)).astype(str)
    right_team_ids = right_possessions.get("team_id", pd.Series(dtype=str)).astype(str)

    for team_id in team_ids:
        left_team_possessions = left_possessions[left_team_ids.eq(team_id)]
        right_team_possessions = right_possessions[right_team_ids.eq(team_id)]
        rows.append(
            {
                "team_id": team_id,
                "left_top5_average": _metric_average(left_team_possessions, left_metric),
                "right_top5_average": _metric_average(right_team_possessions, right_metric),
            }
        )

    summary = pd.DataFrame(rows)
    if summary.empty:
        return summary

    summary["left_average_rank"] = summary["left_top5_average"].rank(
        ascending=False,
        method="min",
        na_option="bottom",
    )
    summary["right_average_rank"] = summary["right_top5_average"].rank(
        ascending=False,
        method="min",
        na_option="bottom",
    )
    summary["combined_average_rank"] = summary[["left_average_rank", "right_average_rank"]].mean(axis=1)
    summary = summary.sort_values(
        ["combined_average_rank", "left_top5_average", "right_top5_average", "team_id"],
        ascending=[True, False, False, True],
    ).reset_index(drop=True)
    summary["browser_rank"] = range(1, len(summary) + 1)
    return summary


def _ranked_team_options(team_summary: pd.DataFrame, value_column: str) -> list[tuple[str, str]]:
    ranked = team_summary.sort_values(
        [value_column, "team_id"],
        ascending=[False, True],
    ).reset_index(drop=True)
    return [
        (f"{rank}. {str(row['team_id']).title()}", str(row["team_id"]))
        for rank, (_, row) in enumerate(ranked.iterrows(), start=1)
    ]


def _team_possessions(possessions: pd.DataFrame, team_id: str) -> pd.DataFrame:
    return possessions[possessions["team_id"].astype(str).eq(team_id)]


def create_team_aec_comparison_browser(
    metric_comparison: dict,
    *,
    left_metric: str = "aec_per_throw",
    right_metric: str = "total_aec",
    left_title: str = "Top 5 by aEC per throw",
    right_title: str = "Top 5 by total aEC",
):
    """Create a notebook browser for team AEC path comparisons."""
    try:
        import ipywidgets as widgets
        from IPython.display import display
    except ImportError as exc:
        raise ImportError(
            "ipywidgets and IPython are required for create_team_aec_comparison_browser."
        ) from exc

    by_metric = metric_comparison["by_metric"]
    middle_by_metric = metric_comparison.get("middle_by_metric", {})
    include_hucks_by_metric = metric_comparison.get("include_hucks_by_metric", {})
    include_hucks_middle_by_metric = metric_comparison.get("include_hucks_middle_by_metric", {})

    def build_view(
        top_by_metric: dict[str, tuple[pd.DataFrame, dict[str, list[pd.DataFrame]]]],
        middle_sets_by_metric: dict[str, tuple[pd.DataFrame, dict[str, list[pd.DataFrame]]]],
        *,
        title_filter_label: str,
    ) -> dict[str, object]:
        left_possessions, left_paths_by_team = top_by_metric[left_metric]
        right_possessions, right_paths_by_team = top_by_metric[right_metric]
        left_middle_possessions, left_middle_paths_by_team = middle_sets_by_metric.get(
            left_metric,
            (pd.DataFrame(), {}),
        )
        right_middle_possessions, right_middle_paths_by_team = middle_sets_by_metric.get(
            right_metric,
            (pd.DataFrame(), {}),
        )
        all_team_ids = sorted(
            set(left_possessions["team_id"].dropna().astype(str))
            | set(right_possessions["team_id"].dropna().astype(str))
        )
        team_summary = _team_top_average_summary(
            all_team_ids,
            left_possessions,
            right_possessions,
            left_metric,
            right_metric,
        )
        if team_summary.empty:
            left_metric_team_options: list[tuple[str, str]] = []
            right_metric_team_options: list[tuple[str, str]] = []
        else:
            left_metric_team_options = _ranked_team_options(team_summary, "left_top5_average")
            right_metric_team_options = _ranked_team_options(team_summary, "right_top5_average")
        return {
            "entries": {
                left_metric: {
                    "top_possessions": left_possessions,
                    "top_paths_by_team": left_paths_by_team,
                    "middle_possessions": left_middle_possessions,
                    "middle_paths_by_team": left_middle_paths_by_team,
                    "team_options": left_metric_team_options,
                },
                right_metric: {
                    "top_possessions": right_possessions,
                    "top_paths_by_team": right_paths_by_team,
                    "middle_possessions": right_middle_possessions,
                    "middle_paths_by_team": right_middle_paths_by_team,
                    "team_options": right_metric_team_options,
                },
            },
            "team_summary": team_summary,
            "title_filter_label": title_filter_label,
        }

    views = {
        "exclude": build_view(
            by_metric,
            middle_by_metric,
            title_filter_label="non-huck",
        )
    }
    if include_hucks_by_metric:
        views["include"] = build_view(
            include_hucks_by_metric,
            include_hucks_middle_by_metric,
            title_filter_label="including hucks",
        )
    has_huck_toggle = "include" in views

    first_view = views["exclude"]
    first_options = first_view["entries"][left_metric]["team_options"]
    if not first_options:
        return widgets.HTML("<b>No qualifying team possessions are available.</b>")

    def make_huck_toggle():
        return widgets.ToggleButtons(
            options=[("No hucks", "exclude"), ("Include hucks", "include")],
            value="exclude",
            description="Hucks",
            layout=widgets.Layout(width="280px"),
            style={"description_width": "54px"},
        )

    def active_huck_key(huck_toggle) -> str:
        return huck_toggle.value if has_huck_toggle else "exclude"

    def metric_options(view_key: str, metric: str) -> list[tuple[str, str]]:
        return views[view_key]["entries"][metric]["team_options"]

    def title_for(team_id: str, window_label: str, view_key: str) -> str:
        filter_label = views[view_key]["title_filter_label"]
        if filter_label == "non-huck":
            return f"{team_id.title()} {window_label.lower()} non-huck long-field scoring possessions"
        return f"{team_id.title()} {window_label.lower()} long-field scoring possessions including hucks"

    def comparison_title(window_label: str, view_key: str, metric_label: str) -> str:
        filter_label = views[view_key]["title_filter_label"]
        if filter_label == "non-huck":
            return f"Team {window_label.lower()} non-huck long-field {metric_label} comparison"
        return f"Team {window_label.lower()} long-field {metric_label} comparison including hucks"

    def make_metric_comparison_tab():
        huck_toggle = make_huck_toggle()
        initial_options = metric_options("exclude", left_metric)
        metric_team_ids = [team_id for _, team_id in initial_options]
        team_dropdown = widgets.Dropdown(
            options=initial_options,
            value=metric_team_ids[0],
            description="Team",
            layout=widgets.Layout(width="320px"),
            style={"description_width": "54px"},
        )
        previous_button = widgets.Button(
            description="Previous",
            icon="chevron-left",
            layout=widgets.Layout(width="112px"),
        )
        next_button = widgets.Button(
            description="Next",
            icon="chevron-right",
            layout=widgets.Layout(width="112px"),
        )
        count_label = widgets.HTML()
        output = widgets.Output()

        def render_team(team_id: str, view_key: str):
            view = views[view_key]
            left_entry = view["entries"][left_metric]
            right_entry = view["entries"][right_metric]
            left_team_possessions = _team_possessions(left_entry["top_possessions"], team_id)
            right_team_possessions = _team_possessions(right_entry["top_possessions"], team_id)
            left_labeled_paths = _label_metric_paths(
                left_team_possessions,
                left_entry["top_paths_by_team"].get(team_id, []),
                left_metric,
                "aEC/T",
            )
            right_labeled_paths = _label_metric_paths(
                right_team_possessions,
                right_entry["top_paths_by_team"].get(team_id, []),
                right_metric,
                "Tot",
            )
            fig = plot_side_by_side_paths(
                left_labeled_paths,
                right_labeled_paths,
                left_title=left_title,
                right_title=right_title,
                title=title_for(team_id, "Top 5", view_key),
                left_summary=_format_metric_average(
                    _metric_average(left_team_possessions, left_metric),
                    left_metric,
                ),
                right_summary=_format_metric_average(
                    _metric_average(right_team_possessions, right_metric),
                    right_metric,
                ),
            )
            with output:
                output.clear_output(wait=True)
                display(fig)

        def update(team_id: str | None = None):
            view_key = active_huck_key(huck_toggle)
            active_options = metric_options(view_key, left_metric)
            metric_team_ids = [team_id for _, team_id in active_options]
            metric_rank_lookup = {
                team_id: rank for rank, (_, team_id) in enumerate(active_options, start=1)
            }
            if team_id is None:
                team_id = team_dropdown.value
            if team_id not in metric_team_ids:
                team_id = metric_team_ids[0]
                team_dropdown.value = team_id
            rank = metric_rank_lookup[team_id]
            count_label.value = f"<b>{rank}</b> of <b>{len(metric_team_ids)}</b>"
            render_team(team_id, view_key)

        def sync_huck_options(_=None):
            view_key = active_huck_key(huck_toggle)
            active_options = metric_options(view_key, left_metric)
            active_ids = [team_id for _, team_id in active_options]
            current_team_id = team_dropdown.value
            team_dropdown.options = active_options
            team_dropdown.value = current_team_id if current_team_id in active_ids else active_ids[0]
            update(team_dropdown.value)

        def on_team_change(change):
            if change["name"] == "value" and change["new"] is not None:
                update(change["new"])

        def on_previous(_):
            metric_team_ids = [team_id for _, team_id in metric_options(active_huck_key(huck_toggle), left_metric)]
            index = metric_team_ids.index(team_dropdown.value)
            team_dropdown.value = metric_team_ids[max(0, index - 1)]

        def on_next(_):
            metric_team_ids = [team_id for _, team_id in metric_options(active_huck_key(huck_toggle), left_metric)]
            index = metric_team_ids.index(team_dropdown.value)
            team_dropdown.value = metric_team_ids[min(len(metric_team_ids) - 1, index + 1)]

        team_dropdown.observe(on_team_change, names="value")
        if has_huck_toggle:
            huck_toggle.observe(sync_huck_options, names="value")
        previous_button.on_click(on_previous)
        next_button.on_click(on_next)
        update(team_dropdown.value)

        control_children = [team_dropdown, previous_button, next_button, count_label]
        if has_huck_toggle:
            control_children.append(huck_toggle)
        controls = widgets.HBox(
            control_children,
            layout=widgets.Layout(align_items="center"),
        )
        return widgets.VBox([controls, output])

    def make_team_comparison_tab(
        *,
        metric: str,
        path_prefix: str,
        panel_suffix: str,
        metric_label: str,
        allow_middle: bool = False,
    ):
        huck_toggle = make_huck_toggle()
        initial_options = metric_options("exclude", metric)
        left_team_dropdown = widgets.Dropdown(
            options=initial_options,
            value=initial_options[0][1],
            description="Left",
            layout=widgets.Layout(width="320px"),
            style={"description_width": "54px"},
        )
        right_default = initial_options[1][1] if len(initial_options) > 1 else initial_options[0][1]
        right_team_dropdown = widgets.Dropdown(
            options=initial_options,
            value=right_default,
            description="Right",
            layout=widgets.Layout(width="320px"),
            style={"description_width": "54px"},
        )
        output = widgets.Output()
        set_toggle = widgets.ToggleButtons(
            options=[("Top 5", "top"), ("Middle 5", "middle")],
            value="top",
            description="Set",
            layout=widgets.Layout(width="220px"),
            style={"description_width": "42px"},
        )

        def team_title(team_id: str) -> str:
            label_lookup = {team_id: label for label, team_id in metric_options(active_huck_key(huck_toggle), metric)}
            return label_lookup[team_id]

        def has_middle_selection(view_key: str) -> bool:
            middle_possessions = views[view_key]["entries"][metric]["middle_possessions"]
            return allow_middle and not middle_possessions.empty

        def active_selection() -> tuple[pd.DataFrame, dict[str, list[pd.DataFrame]], str, str]:
            view_key = active_huck_key(huck_toggle)
            entry = views[view_key]["entries"][metric]
            if has_middle_selection(view_key) and set_toggle.value == "middle":
                return entry["middle_possessions"], entry["middle_paths_by_team"], "Middle 5", view_key
            return entry["top_possessions"], entry["top_paths_by_team"], "Top 5", view_key

        def render_pair(left_team_id: str, right_team_id: str):
            active_possessions, active_paths_by_team, window_label, view_key = active_selection()
            left_team_possessions = _team_possessions(active_possessions, left_team_id)
            right_team_possessions = _team_possessions(active_possessions, right_team_id)
            left_labeled_paths = _label_metric_paths(
                left_team_possessions,
                active_paths_by_team.get(left_team_id, []),
                metric,
                path_prefix,
            )
            right_labeled_paths = _label_metric_paths(
                right_team_possessions,
                active_paths_by_team.get(right_team_id, []),
                metric,
                path_prefix,
            )
            active_panel_suffix = panel_suffix.replace("Top 5", window_label).replace(
                "top 5",
                window_label.lower(),
            )
            fig = plot_side_by_side_paths(
                left_labeled_paths,
                right_labeled_paths,
                left_title=f"{team_title(left_team_id)} {active_panel_suffix}",
                right_title=f"{team_title(right_team_id)} {active_panel_suffix}",
                title=comparison_title(window_label, view_key, metric_label),
                left_summary=_format_metric_average(
                    _metric_average(left_team_possessions, metric),
                    metric,
                    window_label,
                ),
                right_summary=_format_metric_average(
                    _metric_average(right_team_possessions, metric),
                    metric,
                    window_label,
                ),
            )
            with output:
                output.clear_output(wait=True)
                display(fig)

        def update(_=None):
            render_pair(left_team_dropdown.value, right_team_dropdown.value)

        def sync_options(_=None):
            view_key = active_huck_key(huck_toggle)
            active_options = metric_options(view_key, metric)
            active_ids = [team_id for _, team_id in active_options]
            current_left = left_team_dropdown.value
            current_right = right_team_dropdown.value
            left_team_dropdown.options = active_options
            right_team_dropdown.options = active_options
            left_team_dropdown.value = current_left if current_left in active_ids else active_ids[0]
            if current_right in active_ids:
                right_team_dropdown.value = current_right
            else:
                right_team_dropdown.value = active_ids[1] if len(active_ids) > 1 else active_ids[0]
            update()

        left_team_dropdown.observe(update, names="value")
        right_team_dropdown.observe(update, names="value")
        if allow_middle:
            set_toggle.observe(update, names="value")
        if has_huck_toggle:
            huck_toggle.observe(sync_options, names="value")
        update()

        control_children = [left_team_dropdown, right_team_dropdown]
        if allow_middle:
            control_children.append(set_toggle)
        if has_huck_toggle:
            control_children.append(huck_toggle)
        controls = widgets.HBox(
            control_children,
            layout=widgets.Layout(align_items="center"),
        )
        return widgets.VBox([controls, output])

    tabs = widgets.Tab(
        children=[
            make_metric_comparison_tab(),
            make_team_comparison_tab(
                metric=left_metric,
                path_prefix="aEC/T",
                panel_suffix="top 5 by aEC/T",
                metric_label="aEC/T",
                allow_middle=True,
            ),
            make_team_comparison_tab(
                metric=right_metric,
                path_prefix="Tot",
                panel_suffix="top 5 by total aEC",
                metric_label="total aEC",
            ),
        ]
    )
    tabs.set_title(0, "Team metrics")
    tabs.set_title(1, "Compare aEC/T")
    tabs.set_title(2, "Compare total aEC")
    return tabs
