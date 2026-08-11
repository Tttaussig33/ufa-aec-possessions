from __future__ import annotations

from pathlib import Path
import time
from typing import Iterable

import pandas as pd
import requests

BASE_URL = "https://shownspace.com"
USER_AGENT = "ufa-aec-possessions educational analysis"


def _filter_regular_season_games(games: pd.DataFrame) -> pd.DataFrame:
    """Keep games marked as regular season by Shown Space schedule metadata."""
    if games.empty:
        return games

    has_regular_week = "RegularSeasonWeek" in games
    has_playoff_round = "PlayoffRound" in games
    if not has_regular_week and not has_playoff_round:
        return games

    if has_regular_week:
        regular_week = pd.to_numeric(games["RegularSeasonWeek"], errors="coerce")
        regular_mask = regular_week.notna()
    else:
        regular_mask = pd.Series(True, index=games.index)

    if has_playoff_round:
        playoff_round = (
            games["PlayoffRound"]
            .fillna("")
            .astype(str)
            .str.strip()
            .str.lower()
        )
        regular_mask &= playoff_round.isin({"", "none", "nan", "null"})

    return games[regular_mask].reset_index(drop=True)


def fetch_shownspace_games(
    season: int = 2026,
    final_only: bool = True,
    limit: int = 500,
    timeout: int = 30,
    regular_season_only: bool = True,
) -> pd.DataFrame:
    """Fetch Shown Space games, limited to completed regular-season games by default."""
    response = requests.get(
        f"{BASE_URL}/api/games",
        params={"year": season, "limit": limit},
        timeout=timeout,
        headers={"User-Agent": USER_AGENT},
    )
    response.raise_for_status()
    games = pd.DataFrame(response.json().get("games") or [])
    if games.empty:
        return games

    if final_only:
        status = games.get("Status", pd.Series("", index=games.index)).fillna("")
        is_final = games.get("is_final", pd.Series(False, index=games.index)).fillna(False)
        final_mask = is_final.astype(bool) | status.astype(str).str.strip().str.lower().str.startswith("final")
        games = games[final_mask].reset_index(drop=True)

    if regular_season_only:
        games = _filter_regular_season_games(games)
    return games.reset_index(drop=True)


def fetch_shownspace_game_data(game_id: str, timeout: int = 30) -> dict:
    """Fetch one Shown Space game payload."""
    response = requests.get(
        f"{BASE_URL}/api/games/{game_id}",
        timeout=timeout,
        headers={"User-Agent": USER_AGENT},
    )
    response.raise_for_status()
    return response.json()


def fetch_shownspace_throws_for_games(
    game_ids: Iterable[str],
    delay: float = 0.15,
    timeout: int = 30,
) -> pd.DataFrame:
    """Fetch and concatenate throw rows for a list of Shown Space game ids."""
    frames: list[pd.DataFrame] = []
    game_id_list = list(game_ids)
    for index, game_id in enumerate(game_id_list):
        payload = fetch_shownspace_game_data(str(game_id), timeout=timeout)
        throws = pd.DataFrame(payload.get("throws") or [])
        if not throws.empty:
            game = payload.get("game") or {}
            throws["game_id"] = str(game_id)
            if "GameID" not in throws:
                throws["GameID"] = str(game_id)
            throws["home_team_id"] = game.get("HomeTeamID")
            throws["away_team_id"] = game.get("AwayTeamID")
            throws["home_score_final"] = game.get("HomeScore")
            throws["away_score_final"] = game.get("AwayScore")
            throws["start_timestamp"] = game.get("StartTimestamp")
            frames.append(throws)
        if delay > 0 and index < len(game_id_list) - 1:
            time.sleep(delay)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def fetch_shownspace_season_throws(
    season: int = 2026,
    team_id: str | None = None,
    max_games: int | None = None,
    final_only: bool = True,
    delay: float = 0.15,
    regular_season_only: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch season games and throws, using regular-season games by default."""
    games = fetch_shownspace_games(
        season=season,
        final_only=final_only,
        regular_season_only=regular_season_only,
    )
    if team_id is not None and not games.empty:
        team = team_id.lower()
        games = games[
            games["HomeTeamID"].astype(str).str.lower().eq(team)
            | games["AwayTeamID"].astype(str).str.lower().eq(team)
        ].reset_index(drop=True)
    if max_games is not None:
        games = games.head(max_games).reset_index(drop=True)

    throws = fetch_shownspace_throws_for_games(games["GameID"].tolist(), delay=delay)
    return games, throws


def _cache_key(
    season: int,
    team_id: str | None,
    max_games: int | None,
    final_only: bool,
    regular_season_only: bool,
) -> str:
    team_slug = str(team_id).lower() if team_id is not None else "all"
    max_games_slug = str(max_games) if max_games is not None else "all"
    final_slug = "final" if final_only else "all-statuses"
    season_scope_slug = "regular-season" if regular_season_only else "including-playoffs"
    return f"shownspace_throws_{season}_{team_slug}_max-{max_games_slug}_{final_slug}_{season_scope_slug}"


def fetch_shownspace_season_throws_cached(
    season: int = 2026,
    team_id: str | None = None,
    max_games: int | None = None,
    final_only: bool = True,
    delay: float = 0.0,
    cache_dir: str | Path = "data/cache/shownspace",
    force_refresh: bool = False,
    regular_season_only: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch season throws with a cache; refresh from Shown Space when requested."""
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    key = _cache_key(
        season,
        team_id,
        max_games,
        final_only,
        regular_season_only,
    )
    games_path = cache_path / f"{key}_games.pkl"
    throws_path = cache_path / f"{key}_throws.pkl"

    if not force_refresh and games_path.exists() and throws_path.exists():
        return pd.read_pickle(games_path), pd.read_pickle(throws_path)

    games, throws = fetch_shownspace_season_throws(
        season=season,
        team_id=team_id,
        max_games=max_games,
        final_only=final_only,
        delay=delay,
        regular_season_only=regular_season_only,
    )
    games.to_pickle(games_path)
    throws.to_pickle(throws_path)
    return games, throws
