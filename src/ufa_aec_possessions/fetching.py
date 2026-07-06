from __future__ import annotations

import time
from typing import Iterable

import pandas as pd
import requests

BASE_URL = "https://shownspace.com"
USER_AGENT = "ufa-aec-possessions educational analysis"


def fetch_shownspace_games(
    season: int = 2026,
    final_only: bool = True,
    limit: int = 500,
    timeout: int = 30,
) -> pd.DataFrame:
    """Fetch Shown Space game metadata for a season."""
    response = requests.get(
        f"{BASE_URL}/api/games",
        params={"year": season, "limit": limit},
        timeout=timeout,
        headers={"User-Agent": USER_AGENT},
    )
    response.raise_for_status()
    games = pd.DataFrame(response.json().get("games") or [])
    if games.empty or not final_only:
        return games

    status = games.get("Status", pd.Series("", index=games.index)).fillna("")
    is_final = games.get("is_final", pd.Series(False, index=games.index)).fillna(False)
    final_mask = is_final.astype(bool) | status.astype(str).str.strip().str.lower().str.startswith("final")
    return games[final_mask].reset_index(drop=True)


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
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch season games and throw rows, optionally limited to one team."""
    games = fetch_shownspace_games(season=season, final_only=final_only)
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
