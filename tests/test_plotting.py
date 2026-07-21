from __future__ import annotations

import pandas as pd

from ufa_aec_possessions.plotting import (
    plot_possession_path,
    plot_side_by_side_paths,
    plot_team_throw_count_distribution,
    render_shownspace_possession_svg,
)


def test_plotting_smoke():
    path = pd.DataFrame(
        {
            "possession_id": ["p1", "p1"],
            "possession_throw": [1, 2],
            "Thrower": ["a", "b"],
            "Receiver": ["b", "c"],
            "ThrowerX": [0, 5],
            "ThrowerY": [30, 60],
            "ReceiverX": [5, 0],
            "ReceiverY": [60, 105],
            "aec": [0.2, 0.4],
            "cp": [0.95, 0.90],
            "throw_distance": [30.4, 45.3],
        }
    )

    fig = plot_possession_path(path)
    assert len(fig.data) == 1
    svg = render_shownspace_possession_svg(path)
    assert "<svg" in svg
    assert "aEC 0.200" in svg


def test_side_by_side_plotting_smoke():
    path = pd.DataFrame(
        {
            "possession_id": ["p1", "p1"],
            "possession_throw": [1, 2],
            "Thrower": ["a", "b"],
            "Receiver": ["b", "c"],
            "ThrowerX": [0, 5],
            "ThrowerY": [30, 60],
            "ReceiverX": [5, 0],
            "ReceiverY": [60, 105],
            "aec": [0.2, 0.4],
            "cp": [0.95, 0.90],
            "throw_distance": [30.4, 45.3],
        }
    )

    fig = plot_side_by_side_paths(
        {f"aEC/T {rank}: {0.300 - rank / 1000:.3f}": path for rank in range(1, 6)},
        {f"total {rank}: {0.600 - rank / 1000:.3f}": path for rank in range(1, 6)},
    )
    assert len(fig.data) == 10
    assert fig.layout.title.text == "Top scoring possessions"
    assert fig.layout.title.x == 0.5
    assert fig.data[0].showlegend is False
    left_footer = next(annotation.text for annotation in fig.layout.annotations if "<b>aEC per throw</b>" in annotation.text)
    right_footer = next(annotation.text for annotation in fig.layout.annotations if "<b>Total aEC</b>" in annotation.text)
    assert "aEC/T" not in left_footer
    assert "total" not in right_footer
    assert left_footer.count("<br>") == 1
    assert right_footer.count("<br>") == 1


def test_team_throw_count_distribution_uses_team_share():
    possessions = pd.DataFrame(
        {
            "team_id": ["empire", "empire", "empire", "glory", "glory"],
            "possession_id": ["e1", "e2", "e3", "g1", "g2"],
            "throw_count": [5, 5, 8, 3, 7],
            "line_type": ["o_line", "o_line", "o_line", "o_line", "d_line"],
            "outcome": ["goal", "goal", "goal", "goal", "goal"],
        }
    )

    fig = plot_team_throw_count_distribution(possessions)

    assert len(fig.data) == 1
    heatmap = fig.data[0]
    assert list(heatmap.x) == [3, 4, 5, 6, 7, 8]
    assert "Glory (mode 3)" in list(heatmap.y)
    assert "Empire (mode 5)" in list(heatmap.y)
    empire_index = list(heatmap.y).index("Empire (mode 5)")
    throw_five_index = list(heatmap.x).index(5)
    assert heatmap.z[empire_index][throw_five_index] == 2 / 3
