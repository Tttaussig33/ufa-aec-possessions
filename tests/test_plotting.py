from __future__ import annotations

import pandas as pd

from ufa_aec_possessions.plotting import plot_possession_path, render_shownspace_possession_svg


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
