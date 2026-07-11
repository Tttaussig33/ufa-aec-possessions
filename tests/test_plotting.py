from __future__ import annotations

import pandas as pd

from ufa_aec_possessions.plotting import (
    plot_possession_path,
    plot_side_by_side_paths,
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
