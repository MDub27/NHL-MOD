"""Combined V1/V2 totals + separate moneyline report. Explicit invocation only."""
import argparse
import csv
import datetime as dt
import hashlib
import itertools
import json
import pathlib
import platform
import subprocess
from zoneinfo import ZoneInfo
import numpy as np
import sklearn
import nhl_model as v1
import nhl_v2 as v2
from nhl_data import Client, DataError, UTC, atomic_json, collect_season, date_string, history_before, load_xg, schedule, season
from nhl_moneyline import chronological_features, fit, score, backtest


def combinations(selections):
    output = []
    for count in (2, 3):
        for legs in itertools.combinations(selections, count):
            p = [r['probability'] for r in legs]
            same_game = len({r['game_id'] for r in legs}) != count
            output.append({'legs': [r['selection'] for r in legs],
                           'estimated_probability': None if same_game else float(np.prod(p)),
                           'joint_lower_bound': max(0., sum(p) - (count-1)),
                           'joint_upper_bound': min(p),
                           'assumption': 'Same-game dependence: no point estimate' if same_game else 'Independence across distinct games; shared risks not estimated',
                           'market_price': None, 'expected_value': None})
    return output


def build_report(games, history, xg, target, now):
    feature_rows, state = chronological_features(history, xg)
    # Most recent completed season calibrates; earlier seasons train.
    cal_end = f'{season(target)}-07-01'
    train_end = f'{season(target)-1}-07-01'
    fitted = fit(feature_rows, train_end, cal_end)
    total_rows = v2.historical_rows(history, '2024-10-01')
    calibrator = v2.fit_calibrator(total_rows)
    examples, quality = [], []
    for game in games:
        x, q, elo = state.features(game)
        examples.append({'x': x, 'elo': elo})
        quality.append(q)
    scores = score(fitted, examples) if games else None
    output = []
    for i, game in enumerate(games):
        start = dt.datetime.fromisoformat(game['start_utc'].replace('Z', '+00:00'))
        eligible = start > now and game['state'] in {'FUT', 'PRE'}
        warnings = ['Starting goalies unconfirmed; goalie performance, injuries and travel excluded.',
                    'MoneyPuck xG reflects the downloaded historical snapshot, not archived pregame releases.']
        if not eligible:
            warnings.append('Already started/completed or not scheduled to start; date-cutoff reconstruction, not a pregame forecast.')
        q = quality[i]
        if min(q['home_history_games'], q['away_history_games']) < 20:
            warnings.append('Fewer than 20 prior games for at least one team; strong shrinkage toward prior.')
        if q['home_xg_games'] < q['home_history_games'] or q['away_xg_games'] < q['away_history_games']:
            warnings.append('Incomplete xG coverage in recent team history; training-only median imputation used where missing.')
        totals = v1.estimate(game['home'], game['away'], history)
        hp = float(scores['moneyline_calibrated'][i])
        output.append({**game, **totals,
                       'start_pacific': start.astimezone(ZoneInfo('America/Los_Angeles')).isoformat(),
                       'original_over45': totals['probability_over_4_5'],
                       'calibrated_over45': v2.calibrated(totals['probability_over_4_5'], calibrator),
                       'home_ml_raw': float(scores['moneyline_raw'][i]),
                       'home_ml': hp, 'away_ml': 1-hp,
                       'home_goalie': {'name': None, 'status': 'unconfirmed'},
                       'away_goalie': {'name': None, 'status': 'unconfirmed'},
                       'pregame_eligible': eligible, 'data_quality': q,
                       'confidence': 'provisional; see holdout validation and data warnings', 'warnings': warnings})
    eligible = [r for r in output if r['pregame_eligible']]
    top_over = sorted(eligible, key=lambda r: r['calibrated_over45'], reverse=True)[:5]
    ml = [{'game_id': r['id'], 'team': r['home'] if r['home_ml'] >= .5 else r['away'],
           'probability': max(r['home_ml'], r['away_ml'])} for r in eligible]
    top_ml = sorted(ml, key=lambda r: r['probability'], reverse=True)[:5]
    selections = [{'game_id': r['id'], 'selection': r['id']+':over45', 'probability': r['calibrated_over45']} for r in top_over]
    selections += [{**r, 'selection': r['game_id']+':'+r['team']+':ml'} for r in top_ml]
    combo = combinations(selections)
    independent = sorted((r for r in combo if r['estimated_probability'] is not None), key=lambda r: r['estimated_probability'], reverse=True)
    return {'date': target, 'generated_at_utc': now.isoformat(), 'games': output,
            'top_five_over45': [{'game_id': r['id'], 'away': r['away'], 'home': r['home'], 'probability': r['calibrated_over45']} for r in top_over],
            'top_five_moneyline': top_ml,
            'two_game_combinations': [r for r in independent if len(r['legs']) == 2][:10],
            'three_game_combinations': [r for r in independent if len(r['legs']) == 3][:10],
            'same_game_combinations_bounds_only': [r for r in combo if r['estimated_probability'] is None],
            'totals_calibration': calibrator,
            'moneyline_training': {k:v for k,v in fitted.items() if k not in {'full','basic','calibrator'}},
            'history_games': len(history), 'history_last_date': max(r['date'] for r in history),
            'available_features': ['goals for/against (last 50, shrunk)', 'recent goal differential (10 games)',
                                   'MoneyPuck xG for/against (50) and recent xG differential (10)',
                                   'home intercept', 'days rest (capped at 7)', 'back-to-back indicator'],
            'planned_features': ['verified starting goalies and historical pregame goalie GSAx',
                                 'timestamped injuries', 'validated travel distance/timezones'],
            'research_only': True,
            'warnings': ['Experimental ML candidate has not demonstrated superiority to the goals/rest-only baseline; no automatic model promotion.',
                         'Model probabilities are provisional estimates, not guaranteed win rates.',
                         'No market prices supplied: expected value is not evaluated.',
                         'Totals follow unchanged V1 official final-score convention; verify market overtime/shootout settlement rules.',
                         'Combinations exclude started games. Same-game selections have bounds only; do not multiply their probabilities.'],
            'sources': ['NHL public schedule API', 'MoneyPuck.com team game-by-game xG (credited; non-commercial use)']}


def save_snapshot(report, root):
    stamp = dt.datetime.fromisoformat(report['generated_at_utc']).strftime('%Y%m%dT%H%M%S%fZ')
    path = pathlib.Path(root) / (report['date'] + '_' + stamp)
    path.mkdir(parents=True, exist_ok=False)  # Never overwrite a prior pregame snapshot.
    (path/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    pregame = {**report, 'games': [r for r in report['games'] if r['pregame_eligible']]}
    (path/'pregame.json').write_text(json.dumps(pregame, indent=2, allow_nan=False))
    columns = ['id','date','away','home','start_pacific','expected_goals','original_over45','calibrated_over45','home_ml','away_ml','pregame_eligible']
    with (path/'games.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(report['games'])
    lines = [f'# NHL combined report — {report["date"]}', '', f'Generated: {report["generated_at_utc"]}', '',
             '| Away at home | Start (Pacific) | Expected goals | Original O4.5 | Calibrated O4.5 | Home ML | Away ML | Pregame |',
             '|---|---|---:|---:|---:|---:|---:|---|']
    for r in report['games']:
        lines.append(f'| {r["away"]} at {r["home"]} | {r["start_pacific"]} | {r["expected_goals"]:.2f} | {r["original_over45"]:.1%} | {r["calibrated_over45"]:.1%} | {r["home_ml"]:.1%} | {r["away_ml"]:.1%} | {r["pregame_eligible"]} |')
    lines += ['', 'Goalies: unconfirmed for all games. No goalie, injury or travel inputs are modeled.',
              'Data: NHL API and MoneyPuck.com. No market prices or positive-EV claims.',
              'See report.json for ranked picks, combination bounds, provenance and all data warnings.']
    (path/'report.md').write_text('\n'.join(lines)+'\n')
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--date', required=True, type=date_string)
    parser.add_argument('--mode', choices=['predict','backtest','all'], default='all')
    parser.add_argument('--cache', default='data/cache/v3')
    parser.add_argument('--output', default='output/v3')
    parser.add_argument('--xg-csv', help='Optional previously downloaded MoneyPuck team CSV')
    args = parser.parse_args()
    if season(args.date) < 2026:
        parser.error('This fixed 2023/2024/2025 validation protocol requires a target in season 2026 or later')
    client = Client(args.cache)
    all_games = [g for year in range(2023, season(args.date)+1) for g in collect_season(client, year, args.date)]
    history = history_before(all_games, args.date)
    xg = load_xg(client, args.xg_csv)
    rows, _ = chronological_features(history, xg)
    root = pathlib.Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    validation = None
    if args.mode in {'backtest', 'all'}:
        validation, holdout = backtest(rows)
        atomic_json(root/'backtest_metrics.json', validation)
        atomic_json(root/'backtest_predictions.json', holdout)
        print(json.dumps({'backtest': validation}, indent=2))
    if args.mode in {'predict','all'}:
        upcoming = schedule(client, args.date)
        expected_ids = {g['id'] for g in all_games if g['date'] == args.date}
        if {g['id'] for g in upcoming} != expected_ids:
            raise DataError('Requested-day schedule disagrees with club schedules')
        now = dt.datetime.now(UTC)
        report = build_report(upcoming, history, xg, args.date, now)
        report['holdout_validation'] = validation if validation else {'status': 'not_run_in_this_invocation; consult all/backtest artifact'}
        xg_fingerprint = hashlib.sha256(json.dumps(sorted(xg.values(), key=lambda r:(r['id'],r['team'])), sort_keys=True).encode()).hexdigest()
        report['provenance'] = {'git_commit': subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip(),
                                'python': platform.python_version(), 'numpy': np.__version__, 'sklearn': sklearn.__version__,
                                'xg_compact_sha256': xg_fingerprint, 'cache_events': client.events,
                                'history_cutoff_exclusive': args.date}
        path = save_snapshot(report, root)
        print(json.dumps({'report_path': str(path), 'games': len(report['games']),
                          'pregame_games': sum(r['pregame_eligible'] for r in report['games']),
                          'top_over45': report['top_five_over45'], 'top_ml': report['top_five_moneyline']}, indent=2))


if __name__ == '__main__':
    main()
