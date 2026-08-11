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
    fetching.fetch_shownspace_season_throws_cached(
        season=2026,
        cache_dir=tmp_path,
        force_refresh=True,
    )

    assert calls["count"] == 2
    assert first_games.equals(second_games)
    assert first_throws.equals(second_throws)


def test_fetch_shownspace_games_excludes_playoffs_by_default(monkeypatch):
    payload = {
        "games": [
            {
                "GameID": "regular",
                "Status": "Final",
                "is_final": True,
                "RegularSeasonWeek": 12,
                "PlayoffRound": None,
            },
            {
                "GameID": "playoff",
                "Status": "Final",
                "is_final": True,
                "RegularSeasonWeek": None,
                "PlayoffRound": "Semifinal",
            },
        ]
    }

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return payload

    monkeypatch.setattr(fetching.requests, "get", lambda *args, **kwargs: FakeResponse())

    regular_games = fetching.fetch_shownspace_games()
    all_final_games = fetching.fetch_shownspace_games(regular_season_only=False)

    assert regular_games["GameID"].tolist() == ["regular"]
    assert all_final_games["GameID"].tolist() == ["regular", "playoff"]
