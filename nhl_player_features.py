"""Player-level, as-of-date NHL features. No future-game leakage.

This module builds player scoring baselines; it is NOT yet a complete game-total model.
Use verified game-day lineups, never assume the season roster is tonight's lineup.
"""
import datetime as dt
import json
import math
import statistics
import urllib.request

BASE = "https://api-web.nhle.com/v1"

def get_json(path):
    req = urllib.request.Request(BASE + path, headers={"User-Agent": "NHL-MOD/3.0", "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=25) as response:
        return json.load(response)

def player_games(player_id, season, as_of, fetch=get_json):
    """Regular-season NHL game logs, strictly before as_of (YYYY-MM-DD)."""
    payload = fetch(f"/player/{int(player_id)}/game-log/{int(season)}{int(season)+1}/2")
    rows = payload.get("gameLog", [])
    return sorted((g for g in rows if g.get("gameDate", "") < as_of),
                  key=lambda g: g["gameDate"])

def per_game(rows, field):
    if not rows:
        return None
    return sum(float(r.get(field, 0) or 0) for r in rows) / len(rows)

def player_features(player_id, as_of, lookback_seasons=3, fetch=get_json):
    """Multi-season median rate + current-season and last-10 signals."""
    day = dt.date.fromisoformat(as_of)
    current = day.year if day.month >= 7 else day.year - 1
    years = range(max(2000, current - lookback_seasons), current + 1)
    by_season = {yr: player_games(player_id, yr, as_of, fetch) for yr in years}
    prior = [rows for yr, rows in by_season.items() if yr < current and len(rows) >= 10]
    current_rows = by_season[current]
    def rate(field):
        prior_rates = [per_game(rows, field) for rows in prior]
        historical = statistics.median(prior_rates) if prior_rates else None
        season_rate = per_game(current_rows, field)
        recent = per_game(current_rows[-10:], field)
        if historical is None and season_rate is None:
            return {"prior_season_median": None, "current_season": None,
                    "last_10": None, "blended": None}
        if historical is None:
            blended = season_rate
        elif season_rate is None:
            blended = historical
        else:
            # Prior equivalent of 20 games; recent form is diagnostic, not
            # independently added to avoid double-counting current games.
            weight = len(current_rows) / (len(current_rows) + 20)
            blended = (1 - weight) * historical + weight * season_rate
        return {"prior_season_median": historical, "current_season": season_rate,
                "last_10": recent, "blended": blended}
    return {"player_id": int(player_id), "as_of": as_of,
            "current_season_games": len(current_rows),
            "prior_seasons_used": len(prior),
            "goals_per_game": rate("goals"),
            "shots_per_game": rate("shots"),
            "points_per_game": rate("points"),
            "power_play_points_per_game": rate("powerPlayPoints")}

def lineup_scoring_features(player_ids, as_of, fetch=get_json):
    """Aggregates only explicitly supplied expected/confirmed active skaters.

    Not an independent team-goal forecast: player goals are correlated,
    ice-time dependent and must be reconciled with team-level expected goals.
    """
    if not player_ids:
        raise ValueError("A verified expected lineup of player IDs is required")
    players = [player_features(pid, as_of, fetch=fetch) for pid in player_ids]
    goals = [p["goals_per_game"]["blended"] for p in players]
    return {"as_of": as_of, "player_count": len(players),
            "players_without_data": sum(v is None for v in goals),
            "sum_player_goal_rates": sum(v for v in goals if v is not None),
            "players": players}
