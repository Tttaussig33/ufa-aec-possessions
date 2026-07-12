from __future__ import annotations

import pandas as pd

import ufa_aec_possessions.fetching as fetching


def test_fetch_shownspace_season_throws_cached_reuses_local_pickles(tmp_path, monkeypatch):
    calls = {"count": 0}

    def fake_fetch(**_kwargs):
        calls["count"] += 1
        return (
            pd.DataFrame({"GameID": ["game1"]}),
            pd.DataFrame({"GameID": ["game1"], "aec": [0.25]}),
        )

    monkeypatch.setattr(fetching, "fetch_shownspace_season_throws", fake_fetch)

    first_games, first_throws = fetching.fetch_shownspace_season_throws_cached(
        season=2026,
        cache_dir=tmp_path,
    )
    second_games, second_throws = fetching.fetch_shownspace_season_throws_cached(
        season=2026,
        cache_dir=tmp_path,
    )

    assert calls["count"] == 1
    assert first_games.equals(second_games)
    assert first_throws.equals(second_throws)
