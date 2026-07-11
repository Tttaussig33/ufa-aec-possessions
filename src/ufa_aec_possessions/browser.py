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


def create_team_aec_comparison_browser(
    metric_comparison: dict,
    *,
    left_metric: str = "aec_per_throw",
    right_metric: str = "total_aec",
    left_title: str = "Top 5 by aEC per throw",
    right_title: str = "Top 5 by total aEC",
):
    """Create a small notebook browser for flipping through team AEC comparisons."""
    try:
        import ipywidgets as widgets
        from IPython.display import display
    except ImportError as exc:
        raise ImportError(
            "ipywidgets and IPython are required for create_team_aec_comparison_browser."
        ) from exc

    by_metric = metric_comparison["by_metric"]
    left_possessions, left_paths_by_team = by_metric[left_metric]
    right_possessions, right_paths_by_team = by_metric[right_metric]

    team_ids = sorted(
        set(left_possessions["team_id"].dropna().astype(str))
        | set(right_possessions["team_id"].dropna().astype(str))
    )
    if not team_ids:
        return widgets.HTML("<b>No qualifying team possessions are available.</b>")

    team_dropdown = widgets.Dropdown(
        options=[(team_id.title(), team_id) for team_id in team_ids],
        value=team_ids[0],
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

    def render_team(team_id: str):
        left_team_possessions = left_possessions[
            left_possessions["team_id"].astype(str).eq(team_id)
        ]
        right_team_possessions = right_possessions[
            right_possessions["team_id"].astype(str).eq(team_id)
        ]
        left_labeled_paths = _label_metric_paths(
            left_team_possessions,
            left_paths_by_team.get(team_id, []),
            left_metric,
            "aEC/T",
        )
        right_labeled_paths = _label_metric_paths(
            right_team_possessions,
            right_paths_by_team.get(team_id, []),
            right_metric,
            "Tot",
        )
        fig = plot_side_by_side_paths(
            left_labeled_paths,
            right_labeled_paths,
            left_title=left_title,
            right_title=right_title,
            title=f"{team_id.title()} top non-huck long-field scoring possessions",
        )
        with output:
            output.clear_output(wait=True)
            display(fig)

    def update(team_id: str):
        index = team_ids.index(team_id)
        count_label.value = f"<b>{index + 1}</b> of <b>{len(team_ids)}</b>"
        render_team(team_id)

    def on_team_change(change):
        if change["name"] == "value" and change["new"] is not None:
            update(change["new"])

    def on_previous(_):
        index = team_ids.index(team_dropdown.value)
        team_dropdown.value = team_ids[max(0, index - 1)]

    def on_next(_):
        index = team_ids.index(team_dropdown.value)
        team_dropdown.value = team_ids[min(len(team_ids) - 1, index + 1)]

    team_dropdown.observe(on_team_change, names="value")
    previous_button.on_click(on_previous)
    next_button.on_click(on_next)
    update(team_dropdown.value)

    controls = widgets.HBox(
        [team_dropdown, previous_button, next_button, count_label],
        layout=widgets.Layout(align_items="center"),
    )
    return widgets.VBox([controls, output])
