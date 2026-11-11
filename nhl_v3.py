"""NHL 4.5 V3 experimental player-adjusted Monte Carlo engine.

Explicit, date-stamped inputs are required for lineup and market data.
No live-lineup or sportsbook feed is implied by this module.
"""
import argparse
import json
import math
import pathlib
import random
import statistics
from nhl_model import estimate

def poisson(rng, lam):
    if not 0 <= lam <= 20:
        raise ValueError("Expected goals must be between 0 and 20")
    # Inverse CDF, efficient at NHL scoring rates.
    u = rng.random()
    p = math.exp(-lam)
    cumulative = p
    n = 0
    while u > cumulative:
        n += 1
        p *= lam / n
        cumulative += p
        if n > 100:
            raise ArithmeticError("Poisson sampler did not converge")
    return n

def grade_player(player):
    """Comparable-player grade from pregame per-game stats, 0–100.

    Descriptive ranking only; not a fitted predictive coefficient.
    """
    goals = max(0.0, float(player.get("goals_per_game", 0)))
    shots = max(0.0, float(player.get("shots_per_game", 0)))
    toi = max(0.0, float(player.get("toi_minutes", 0)))
    pp = max(0.0, float(player.get("power_play_points_per_game", 0)))
    return round(min(100, 40*min(goals/0.6,1) + 25*min(shots/4,1)
                     + 20*min(toi/22,1) + 15*min(pp/0.6,1)),2)

def comparable_players(target, candidates, limit=5):
    """Nearest peers by normalized rate and usage; position must match."""
    fields = {"goals_per_game":0.6,"shots_per_game":4,
              "toi_minutes":22,"power_play_points_per_game":0.6}
    peers=[]
    for candidate in candidates:
        if candidate.get("id")==target.get("id") or candidate.get("position")!=target.get("position"):
            continue
        distance=math.sqrt(sum(((float(candidate.get(k,0))-float(target.get(k,0)))/scale)**2
                               for k,scale in fields.items()))
        peers.append({"id":candidate.get("id"),"distance":round(distance,4),
                      "grade":grade_player(candidate)})
    return sorted(peers,key=lambda p:p["distance"])[:limit]

def team_multiplier(team):
    """Conservative player adjustment relative to explicitly provided baseline.

    Team input: lineup list, baseline_player_goals_per_game, and goalie adjustment.
    The sum of player rates is NOT used directly as team expected goals.
    """
    lineup=team.get("lineup")
    baseline=team.get("baseline_player_goals_per_game")
    if not isinstance(lineup,list) or not lineup or baseline is None or float(baseline)<=0:
        raise ValueError("Require verified lineup and positive historical lineup baseline")
    if any("goals_per_game" not in p for p in lineup):
        raise ValueError("Each player must have goals_per_game")
    current=sum(max(0,float(p["goals_per_game"])) for p in lineup)
    ratio=max(.75,min(1.25,current/float(baseline)))
    return 1+0.5*(ratio-1)  # shrunk to avoid double-counting team history

def simulate(home_lambda,away_lambda,n=1000,seed=42,dispersion=0.12):
    """Shared per-game pace shock correlates the two team scores.

    A lognormal pace shock with mean 1 captures common high/low tempo;
    not fitted to NHL data yet. Results are exploratory until calibrated.
    """
    if n<=0 or not 0<=dispersion<=0.5:
        raise ValueError("Invalid simulation count or dispersion")
    rng=random.Random(seed)
    games=[]
    for _ in range(n):
        pace=math.exp(rng.gauss(-dispersion**2/2,dispersion))
        h=poisson(rng,home_lambda*pace)
        a=poisson(rng,away_lambda*pace)
        games.append({"home":h,"away":a,"total":h+a,"margin":h-a})
    return games

def evaluate_market(games,book_line,over_decimal=None,under_decimal=None,fee_fraction=0):
    """Two-goal teasers: over(line-2) or under(line+2).

    For half-goal lines there are no pushes. Decimal odds include returned stake.
    """
    if book_line<=2 or book_line%1!=0.5:
        raise ValueError("Require a half-goal book total greater than 2")
    if not 0<=fee_fraction<1:
        raise ValueError("fee_fraction must be in [0,1)")
    n=len(games)
    if n==0:
        raise ValueError("No simulated games")
    choices=[]
    for name,line,odds,win in (
        ("over",book_line-2,over_decimal,lambda total: total>book_line-2),
        ("under",book_line+2,under_decimal,lambda total: total<book_line+2)):
        p=sum(win(g["total"]) for g in games)/n
        row={"side":name,"teased_line":line,"probability":round(p,4),
             "wins":round(p*n),"simulations":n}
        if odds is not None:
            if odds<=1:raise ValueError("Decimal odds must exceed 1")
            net_profit_per_win=(odds-1)*(1-fee_fraction)
            ev=p*net_profit_per_win-(1-p)
            row.update(decimal_odds=odds,expected_net_return_per_unit=round(ev,5),
                       break_even_probability=round(1/(1+net_profit_per_win),5))
        choices.append(row)
    return choices

def analyze(payload):
    if payload.get("as_of") is None or payload.get("book_total") is None:
        raise ValueError("Need as_of date and observed sportsbook total")
    if not payload.get("source_timestamp_utc"):
        raise ValueError("Need source_timestamp_utc for reproducibility")
    home=payload["home"]; away=payload["away"]
    if "team_expected_goals" not in home or "team_expected_goals" not in away:
        raise ValueError("Require team expected goals from an as-of-date V1 baseline")
    hm=team_multiplier(home); am=team_multiplier(away)
    # Opposing goalie effects must be externally sourced; zero is neutral.
    hgoalie=float(away.get("opposing_goalie_effect",0))
    agoalie=float(home.get("opposing_goalie_effect",0))
    if abs(hgoalie)>.25 or abs(agoalie)>.25:
        raise ValueError("Goalie adjustments beyond 25% require refitting")
    hl=float(home["team_expected_goals"])*hm*(1+hgoalie)
    al=float(away["team_expected_goals"])*am*(1+agoalie)
    games=simulate(hl,al,int(payload.get("simulations",1000)),int(payload.get("seed",42)))
    choices=evaluate_market(games,float(payload["book_total"]),
                            payload.get("over_decimal"),payload.get("under_decimal"),
                            float(payload.get("fee_fraction",0)))
    values=[g["total"] for g in games]
    values.sort()
    ranked=sorted((c for c in choices if "expected_net_return_per_unit" in c),
                  key=lambda x:x["expected_net_return_per_unit"],reverse=True)
    return {"as_of":payload["as_of"],"source_timestamp_utc":payload["source_timestamp_utc"],
            "matchup":f'{away["name"]} @ {home["name"]}',
            "expected_goals":{"home":round(hl,3),"away":round(al,3)},
            "simulation":{"count":len(games),"mean_total":round(statistics.mean(values),3),
                          "median_total":statistics.median(values),
                          "central_90pct_total":[values[int(.05*(len(values)-1))],
                                                  values[int(.95*(len(values)-1))]],
                          "minimum":values[0],"maximum":values[-1]},
            "teased_markets":choices,
            "best_value":ranked[0]["side"] if ranked else None,
            "warning":"Exploratory: player coefficients, dispersion and goalie effects are not empirically fitted. Prices and lineup must be verified externally."}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--input",required=True,help="Timestamped game-day JSON")
    p.add_argument("--output",default="output/v3_simulation.json")
    args=p.parse_args()
    data=json.loads(pathlib.Path(args.input).read_text())
    result=analyze(data)
    dest=pathlib.Path(args.output);dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))

if __name__=="__main__":
    main()
