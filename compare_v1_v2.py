"""Leakage-aware rolling evaluation of V1 vs V2 on saved historical rows.

Input is a CSV of chronological out-of-sample V1 predictions from nhl_model.backtest.
The test uses only preceding dates to fit V2 calibration. Same-day outcomes are
never used to fit or rank predictions. This evaluates OVER 4.5 only, NOT teased
market EV, which requires archived game-day odds.
"""
import argparse
import csv
import json
from collections import defaultdict
import nhl_v2

def evaluate(rows, min_train=150, top_k=3):
    rows=sorted(rows,key=lambda r:(r["date"],r["home"],r["away"]))
    days=defaultdict(list)
    for row in rows:
        if not {"date","home","away","probability_over_4_5","actual"}<=row.keys():
            raise ValueError("Missing required CSV fields")
        row=dict(row)
        row["probability_over_4_5"]=float(row["probability_over_4_5"])
        row["actual"]=int(row["actual"])
        if row["actual"] not in (0,1):raise ValueError("Outcome must be 0 or 1")
        days[row["date"]].append(row)
    training=[];scored=[];day_results=[]
    for day, games in sorted(days.items()):
        if len(training)>=min_train:
            params=nhl_v2.fit_calibrator(training)
            preds=[{**g,"calibrated_probability":nhl_v2.calibrated(g["probability_over_4_5"],params)} for g in games]
            for g in preds:g["train_games"]=len(training)
            scored.extend(preds)
            for key in ("probability_over_4_5","calibrated_probability"):
                top=sorted(preds,key=lambda r:(-r[key],r["home"],r["away"]))[:top_k]
                day_results.append({"date":day,"model":"V1" if key=="probability_over_4_5" else "V2",
                                    "games":len(top),"hits":sum(g["actual"] for g in top),
                                    "average_probability":sum(g[key] for g in top)/len(top)})
        training.extend(games)
    if not scored:raise ValueError("Not enough earlier observations for walk-forward evaluation")
    def model_report(key):
        m=nhl_v2.metrics(scored,key)
        top=[r for r in day_results if r["model"]==("V1" if key=="probability_over_4_5" else "V2")]
        m["top_k_hit_rate"]=round(sum(d["hits"] for d in top)/sum(d["games"] for d in top),5)
        m["top_k_games"]=sum(d["games"] for d in top)
        return m
    rate=sum(r["actual"] for r in scored)/len(scored)
    baseline_brier=rate*(1-rate) # hindsight diagnostic, not a deployable pregame baseline
    changed_days=0
    for day in sorted({r["date"] for r in scored}):
        games=[r for r in scored if r["date"]==day]
        v1=[(r["home"],r["away"]) for r in sorted(games,key=lambda r:(-r["probability_over_4_5"],r["home"],r["away"]))[:top_k]]
        v2=[(r["home"],r["away"]) for r in sorted(games,key=lambda r:(-r["calibrated_probability"],r["home"],r["away"]))[:top_k]]
        changed_days+=v1!=v2
    return {"games":len(scored),"days":len({r["date"] for r in scored}),
            "V1":model_report("probability_over_4_5"),
            "V2":model_report("calibrated_probability"),
            "top_k_changed_days":changed_days,
            "hindsight_constant_baseline_brier":round(baseline_brier,6),
            "caution":"V2 sigmoid calibration is monotonic for positive slope and cannot improve ranking; Brier and log loss assess probability quality. Teaser EV cannot be evaluated without historical quoted prices."}

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--input",required=True)
    parser.add_argument("--min-train",type=int,default=150)
    parser.add_argument("--top-k",type=int,default=3)
    args=parser.parse_args()
    with open(args.input,newline="") as f:rows=list(csv.DictReader(f))
    print(json.dumps(evaluate(rows,args.min_train,args.top_k),indent=2))

if __name__=="__main__":main()
