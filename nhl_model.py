"""NHL over-4.5 baseline using public NHL API and standard library."""
import argparse, csv, datetime as dt, json, math, pathlib, urllib.request
from collections import defaultdict
API='https://api-web.nhle.com/v1'
TEAMS='ANA BOS BUF CAR CBJ CGY CHI COL DAL DET EDM FLA LAK MIN MTL NJD NSH NYI NYR OTT PHI PIT SEA SJS STL TBL TOR UTA VAN VGK WPG WSH'.split()
CACHE=pathlib.Path('data/cache'); OUT=pathlib.Path('output')
def fetch(endpoint):
    req=urllib.request.Request(API+endpoint,headers={'User-Agent':'NHL-MOD/1.0','Accept':'application/json'})
    with urllib.request.urlopen(req,timeout=30) as response:return json.load(response)
def season(date):return date.year if date.month>=7 else date.year-1
def collect(start,refresh=False):
    CACHE.mkdir(parents=True,exist_ok=True); path=CACHE/f'{start}.json'
    if path.exists() and not refresh:return json.loads(path.read_text())
    games={}; failures=[]
    for team in TEAMS:
        try: payload=fetch(f'/club-schedule-season/{team}/{start}{start+1}')
        except Exception as exc: failures.append(f'{team}: {exc}');continue
        for g in payload.get('games',[]):
            if g.get('gameType')!=2:continue
            h=g.get('homeTeam') or {};a=g.get('awayTeam') or {}
            if h.get('score') is None or a.get('score') is None:continue
            gid=str(g.get('id','')); date=g.get('gameDate')
            if not gid or not date:continue
            games[gid]={'id':gid,'date':date[:10],'home':h.get('abbrev',''),'away':a.get('abbrev',''),'hg':int(h['score']),'ag':int(a['score'])}
    if not games:raise RuntimeError(f'No completed games retrieved for {start}: {failures[:3]}')
    if failures:print('Unavailable club schedules:',failures)
    rows=sorted(games.values(),key=lambda g:(g['date'],g['id']))
    path.write_text(json.dumps(rows));return rows
def estimate(home,away,history):
    if not history:lam=5.9
    else:
        avg=sum(g['hg']+g['ag'] for g in history)/(2*len(history))
        h_avg=sum(g['hg'] for g in history)/len(history)
        a_avg=sum(g['ag'] for g in history)/len(history)
        def rates(team):
            rows=[g for g in history if team in (g['home'],g['away'])]
            gf=[g['hg'] if g['home']==team else g['ag'] for g in rows]
            ga=[g['ag'] if g['home']==team else g['hg'] for g in rows]
            f=(sum(gf)+15*avg)/(len(gf)+15);a=(sum(ga)+15*avg)/(len(ga)+15)
            return (.8*f+.2*(sum(gf[-10:])/len(gf[-10:]) if gf else f),.8*a+.2*(sum(ga[-10:])/len(ga[-10:]) if ga else a))
        hf,ha=rates(home);af,aa=rates(away)
        lam=((hf+aa)/2)*(h_avg/avg)+((af+ha)/2)*(a_avg/avg)
    lam=max(.1,lam)
    p=1-math.exp(-lam)*sum(lam**k/math.factorial(k) for k in range(5))
    return {'expected_goals':round(lam,3),'probability_over_4_5':round(p,5)}
def backtest(games,start_date):
    history=[];pred=[];days=defaultdict(list)
    for g in games:days[g['date']].append(g)
    for date in sorted(days):
        if date>=start_date and len(history)>=150:
            for g in days[date]:
                e=estimate(g['home'],g['away'],history)
                pred.append({'date':date,'home':g['home'],'away':g['away'],**e,'actual':int(g['hg']+g['ag']>=5)})
        history.extend(days[date])
    return pred
def summarize(rows):
    if not rows:return {'games':0,'warning':'No eligible backtest games'}
    n=len(rows);base=sum(r['actual'] for r in rows)/n
    brier=sum((r['probability_over_4_5']-r['actual'])**2 for r in rows)/n
    loss=-sum(r['actual']*math.log(max(1e-12,r['probability_over_4_5']))+(1-r['actual'])*math.log(max(1e-12,1-r['probability_over_4_5'])) for r in rows)/n
    days=defaultdict(list)
    for r in rows:days[r['date']].append(r)
    top=[r for group in days.values() for r in sorted(group,key=lambda x:x['probability_over_4_5'],reverse=True)[:3]]
    return {'games':n,'actual_over_rate':round(base,4),'brier':round(brier,5),'constant_baseline_brier':round(base*(1-base),5),'log_loss':round(loss,5),'top_three_game_hit_rate':round(sum(r['actual'] for r in top)/len(top),4)}
def upcoming(date):
    data=fetch('/schedule/'+date);rows=[]
    for day in data.get('gameWeek',[]):
        if day.get('date')!=date:continue
        for g in day.get('games',[]):
            if g.get('gameType')!=2 or g.get('gameState') in ('FINAL','OFF'):continue
            rows.append({'date':date,'home':(g.get('homeTeam') or {}).get('abbrev',''),'away':(g.get('awayTeam') or {}).get('abbrev',''),'start_utc':g.get('startTimeUTC')})
    return rows
def write_csv(name,rows):
    if rows:
        with (OUT/name).open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
def main():
    p=argparse.ArgumentParser();p.add_argument('--date',default=dt.date.today().isoformat());p.add_argument('--mode',choices=['all','predict','backtest'],default='all');p.add_argument('--first-season',type=int,default=2023);p.add_argument('--test-from',default='2025-10-01');p.add_argument('--refresh',action='store_true');args=p.parse_args()
    date=dt.date.fromisoformat(args.date);OUT.mkdir(exist_ok=True)
    games={g['id']:g for start in range(args.first_season,season(date)+1) for g in collect(start,args.refresh)}
    history=sorted((g for g in games.values() if g['date']<args.date),key=lambda g:(g['date'],g['id']))
    if args.mode in ('all','backtest'):
        rows=backtest(history,args.test_from);summary=summarize(rows)
        write_csv('backtest_predictions.csv',rows);(OUT/'backtest_metrics.json').write_text(json.dumps(summary,indent=2));print('Backtest:',summary)
    if args.mode in ('all','predict'):
        rows=sorted(({**g,**estimate(g['home'],g['away'],history)} for g in upcoming(args.date)),key=lambda g:g['probability_over_4_5'],reverse=True)
        write_csv('daily_predictions.csv',rows);(OUT/'daily_predictions.json').write_text(json.dumps({'date':args.date,'predictions':rows},indent=2));print('Upcoming:',rows[:3])
if __name__=='__main__':main()
