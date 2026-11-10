"""NHL 4.5 V2: chronological calibration, date-specific predictions and evaluation.
Uses the original nhl_model.py estimator unchanged.
"""
import argparse
import csv
import datetime as dt
import json
import math
import pathlib
from collections import defaultdict
import nhl_model as v1

OUT = pathlib.Path("output")
EPS = 1e-6

def clamp(p):
    return max(EPS, min(1-EPS, float(p)))

def sigmoid(z):
    if z >= 0:
        return 1/(1+math.exp(-z))
    e=math.exp(z)
    return e/(1+e)

def logit(p):
    p=clamp(p)
    return math.log(p/(1-p))

def fit_calibrator(rows):
    """Regularized Platt scaling on chronological out-of-sample V1 predictions."""
    if len(rows) < 100 or len({r["actual"] for r in rows}) < 2:
        return {"intercept":0.0,"slope":1.0,"trained_games":0,"status":"insufficient_data"}
    x=[logit(r["probability_over_4_5"]) for r in rows]
    y=[int(r["actual"]) for r in rows]
    a,b=0.0,1.0
    for _ in range(60):
        probs=[sigmoid(a+b*t) for t in x]
        g0=sum(p-t for p,t in zip(probs,y))+0.1*a
        g1=sum((p-t)*z for p,t,z in zip(probs,y,x))+0.1*(b-1)
        w=[max(p*(1-p),1e-9) for p in probs]
        h00=sum(w)+0.1
        h01=sum(q*z for q,z in zip(w,x))
        h11=sum(q*z*z for q,z in zip(w,x))+0.1
        det=h00*h11-h01*h01
        if det <= 0: break
        da=(h11*g0-h01*g1)/det
        db=(h00*g1-h01*g0)/det
        scale=min(1.0,2/max(abs(da),abs(db),1e-12))
        a-=scale*da
        b-=scale*db
        if max(abs(da),abs(db)) < 1e-8:break
    return {"intercept":round(a,8),"slope":round(b,8),"trained_games":len(rows),"status":"fitted"}

def calibrated(p,params):
    return round(clamp(sigmoid(params["intercept"]+params["slope"]*logit(p))),5)

def metrics(rows,key):
    if not rows:return {"games":0}
    n=len(rows)
    brier=sum((clamp(r[key])-r["actual"])**2 for r in rows)/n
    loss=-sum(r["actual"]*math.log(clamp(r[key]))+(1-r["actual"])*math.log(1-clamp(r[key])) for r in rows)/n
    return {"games":n,"actual_over_rate":round(sum(r["actual"] for r in rows)/n,5),"mean_probability":round(sum(r[key] for r in rows)/n,5),"brier":round(brier,6),"log_loss":round(loss,6)}

def historical_rows(games,first_date):
    # V1 backtest groups games by date, so games on the same date never inform each other.
    return v1.backtest(games,first_date)

def split_and_fit(rows,train_end,validation_end):
    train=[r for r in rows if r["date"]<train_end]
    validation=[r for r in rows if train_end<=r["date"]<validation_end]
    params=fit_calibrator(train)
    scored=[{**r,"calibrated_probability":calibrated(r["probability_over_4_5"],params)} for r in validation]
    return params,scored

def write_csv(path,rows,columns):
    with path.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=columns,extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--date",default=dt.date.today().isoformat())
    p.add_argument("--mode",choices=["predict","backtest","evaluate"],default="predict")
    p.add_argument("--first-season",type=int,default=2023)
    p.add_argument("--backtest-from",default="2024-10-01")
    p.add_argument("--train-end",default="2025-10-01")
    p.add_argument("--validation-end",default="2026-07-01")
    p.add_argument("--refresh",action="store_true")
    args=p.parse_args()
    target=dt.date.fromisoformat(args.date)
    train_end=dt.date.fromisoformat(args.train_end)
    validation_end=dt.date.fromisoformat(args.validation_end)
    if not dt.date.fromisoformat(args.backtest_from)<train_end<validation_end:
        p.error("Require backtest-from < train-end < validation-end")
    OUT.mkdir(exist_ok=True)
    # Only fetch seasons that could contribute data before the requested date.
    # Scores after the requested date are excluded from training and feature history.
    all_games={}
    for yr in range(args.first_season,v1.season(target)+1):
        for g in v1.collect(yr,args.refresh):
            if g["date"]<=args.date:
                all_games[g["id"]]=g
    history=sorted((g for g in all_games.values() if g["date"]<args.date),key=lambda g:(g["date"],g["id"]))\n    evaluation_history=sorted(all_games.values(),key=lambda g:(g["date"],g["id"]))
    rows=historical_rows(history,args.backtest_from)
    if args.mode=="backtest":
        params,holdout=split_and_fit(rows,args.train_end,args.validation_end)
        report={"calibration":params,"training_games":sum(r["date"]<args.train_end for r in rows),
                "validation_period":{"from":args.train_end,"to_exclusive":args.validation_end},
                "original":metrics(holdout,"probability_over_4_5"),
                "calibrated":metrics(holdout,"calibrated_probability"),
                "note":"Validation only; calibration is not guaranteed to improve results. Do not tune on holdout."}
        write_csv(OUT/"validation_predictions.csv",holdout,
                  ["date","home","away","expected_goals","probability_over_4_5","calibrated_probability","actual"])
        (OUT/"validation_metrics.json").write_text(json.dumps(report,indent=2))
        print(json.dumps(report,indent=2))
        return
    if args.mode=="predict":
        # Fit only on prior completed games; no scores from target day.
        params=fit_calibrator([r for r in rows if r["date"]<args.date])
        scheduled=v1.upcoming(args.date)
        predictions=[]
        for g in scheduled:
            est=v1.estimate(g["home"],g["away"],history)
            predictions.append({**g,**est,"calibrated_probability":calibrated(est["probability_over_4_5"],params)})
        predictions.sort(key=lambda r:r["calibrated_probability"],reverse=True)
        report={"date":args.date,"generated_at_utc":dt.datetime.now(dt.timezone.utc).isoformat(),
                "calibration":params,"predictions":predictions,
                "note":"Run before games begin. Download and retain this artifact; later runs are not an immutable record."}
        write_csv(OUT/"predictions.csv",predictions,
                  ["date","home","away","start_utc","expected_goals","probability_over_4_5","calibrated_probability"])
        (OUT/"predictions.json").write_text(json.dumps(report,indent=2))
        print(json.dumps(report,indent=2))
        return
    # Historical evaluation uses the predictions generated without same-day results.
    # It is NOT a verification of a previously saved live prediction artifact.
    day=[r for r in historical_rows(evaluation_history,args.backtest_from) if r["date"]==args.date]
    params=fit_calibrator([r for r in rows if r["date"]<args.date])
    scored=[{**r,"calibrated_probability":calibrated(r["probability_over_4_5"],params)} for r in day]
    write_csv(OUT/"evaluation.csv",scored,
              ["date","home","away","expected_goals","probability_over_4_5","calibrated_probability","actual"])
    report={"date":args.date,"games_evaluated":len(scored),"original":metrics(scored,"probability_over_4_5"),
            "calibrated":metrics(scored,"calibrated_probability"),
            "note":"Historical reconstruction, not comparison to a saved pregame artifact. Only completed games in cached NHL schedules are included."}
    (OUT/"evaluation.json").write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))

if __name__=="__main__":
    main()
