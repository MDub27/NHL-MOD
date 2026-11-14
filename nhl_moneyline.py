"""Chronological NHL win classifier. Labels include overtime and shootouts."""
from collections import defaultdict, deque
import datetime as dt
import math
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = ['goals_for_diff', 'goals_against_diff', 'recent_goal_diff',
            'xg_for_diff', 'xg_against_diff', 'recent_xg_diff',
            'rest_days_diff', 'back_to_back_diff', 'xg_missing']
GOALS_ONLY = [0, 1, 2, 6, 7]


def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -35, 35)))


class FeatureState:
    def __init__(self):
        self.teams = defaultdict(lambda: deque(maxlen=50))
        self.elo = defaultdict(lambda: 1500.)
        self.games = 0

    def rates(self, club, date):
        rows = self.teams[club]
        gf = (sum(r['gf'] for r in rows) + 15 * 3.) / (len(rows) + 15)
        ga = (sum(r['ga'] for r in rows) + 15 * 3.) / (len(rows) + 15)
        recent = list(rows)[-10:]
        gd = sum(r['gf'] - r['ga'] for r in recent) / max(1, len(recent))
        xrows = [r for r in rows if r['xf'] is not None]
        # Missing xG stays missing; the fitted pipeline imputes it with training medians.
        xf = sum(r['xf'] for r in xrows) / len(xrows) if xrows else np.nan
        xa = sum(r['xa'] for r in xrows) / len(xrows) if xrows else np.nan
        rx = [r for r in recent if r['xf'] is not None]
        xd = sum(r['xf'] - r['xa'] for r in rx) / len(rx) if rx else np.nan
        rest = min(7, (dt.date.fromisoformat(date) - dt.date.fromisoformat(rows[-1]['date'])).days) if rows else 7
        return [gf, ga, gd, xf, xa, xd, rest, int(rest == 1)], len(xrows), len(rows)

    def features(self, game):
        h, hx, hn = self.rates(game['home'], game['date'])
        a, ax, an = self.rates(game['away'], game['date'])
        x = [h[i] - a[i] for i in range(8)]
        x.append(int(hx < hn or ax < an or not hx or not ax))
        elo = 1 / (1 + 10 ** ((self.elo[game['away']] - self.elo[game['home']] - 50) / 400))
        return x, {'home_history_games': hn, 'away_history_games': an,
                   'home_xg_games': hx, 'away_xg_games': ax}, elo

    def add(self, game, xg):
        h, a = game['home'], game['away']
        _, _, p = self.features(game)
        delta = 20 * (int(game['hg'] > game['ag']) - p)
        self.elo[h] += delta
        self.elo[a] -= delta
        for club, gf, ga in [(h, game['hg'], game['ag']), (a, game['ag'], game['hg'])]:
            xr = xg.get((game['id'], club))
            if xr and xr['date'] != game['date']:
                raise ValueError('xG and NHL date mismatch')
            self.teams[club].append({'date': game['date'], 'gf': gf, 'ga': ga,
                                     'xf': xr['xf'] if xr else None, 'xa': xr['xa'] if xr else None})
        self.games += 1


def chronological_features(games, xg, min_history=200):
    state = FeatureState()
    days = defaultdict(list)
    for g in games:
        days[g['date']].append(g)
    rows = []
    for date in sorted(days):
        # Generate the whole day first. Same-day outcomes cannot affect features or Elo.
        if state.games >= min_history:
            for g in sorted(days[date], key=lambda r: r['id']):
                x, quality, elo = state.features(g)
                rows.append({'id': g['id'], 'date': date, 'x': x, 'elo': elo,
                             'y': int(g['hg'] > g['ag']), 'quality': quality})
        for g in sorted(days[date], key=lambda r: r['id']):
            state.add(g, xg)
    return rows, state


def classifier(rows, indices):
    if len(rows) < 200 or len({r['y'] for r in rows}) != 2:
        raise ValueError('Need at least 200 chronological training games with both outcomes')
    model = make_pipeline(SimpleImputer(strategy='median', keep_empty_features=True),
                          StandardScaler(), LogisticRegression(C=1., max_iter=2000, random_state=0))
    model.fit(np.array([r['x'] for r in rows])[:, indices], [r['y'] for r in rows])
    return model


def probabilities(model, rows, indices):
    return model.predict_proba(np.array([r['x'] for r in rows])[:, indices])[:, 1]


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def fit(rows, train_end, calibration_end):
    train = [r for r in rows if r['date'] < train_end]
    cal = [r for r in rows if train_end <= r['date'] < calibration_end]
    if len(cal) < 200 or len({r['y'] for r in cal}) != 2:
        raise ValueError('Need at least 200 separate chronological calibration games')
    full_indices = list(range(len(FEATURES)))
    full, basic = classifier(train, full_indices), classifier(train, GOALS_ONLY)
    calibrator = LogisticRegression(C=1., max_iter=2000, random_state=0)
    calibrator.fit(logit(probabilities(full, cal, full_indices)).reshape(-1, 1), [r['y'] for r in cal])
    return {'full': full, 'basic': basic, 'calibrator': calibrator,
            'home_prior': (sum(r['y'] for r in train) + 1) / (len(train) + 2),
            'train_games': len(train), 'calibration_games': len(cal),
            'train_end_exclusive': train_end, 'calibration_end_exclusive': calibration_end}


def score(fitted, rows):
    p = np.clip(probabilities(fitted['full'], rows, list(range(len(FEATURES)))), 1e-6, 1-1e-6)
    calibrated = np.clip(fitted['calibrator'].predict_proba(logit(p).reshape(-1, 1))[:, 1], 1e-6, 1-1e-6)
    return {'moneyline_raw': p, 'moneyline_calibrated': calibrated,
            'goals_only': probabilities(fitted['basic'], rows, GOALS_ONLY),
            'home_rate': np.repeat(fitted['home_prior'], len(rows)),
            'coin_flip': np.repeat(.5, len(rows)), 'elo': np.array([r['elo'] for r in rows])}


def metrics(y, p):
    y, p = np.array(y), np.clip(np.array(p), 1e-6, 1 - 1e-6)
    bins = []
    for lo in np.arange(0, 1, .1):
        mask = (p >= lo) & (p < lo + .1)
        if mask.any():
            bins.append({'from': round(float(lo), 1), 'to': round(float(lo + .1), 1),
                         'games': int(mask.sum()), 'mean_probability': float(p[mask].mean()),
                         'actual_home_win_rate': float(y[mask].mean())})
    return {'games': len(y), 'brier': float(np.mean((p-y)**2)),
            'log_loss': float(-np.mean(y*np.log(p)+(1-y)*np.log(1-p))),
            'accuracy': float(np.mean((p >= .5) == y)),
            'mean_probability': float(p.mean()), 'actual_home_win_rate': float(y.mean()),
            'expected_calibration_error': sum(b['games'] * abs(b['mean_probability'] - b['actual_home_win_rate']) for b in bins) / len(y),
            'calibration_bins': bins}


def backtest(rows):
    # Fixed protocol decided before looking at scores: 2023 train, 2024 calibrate, 2025 test.
    fitted = fit(rows, '2024-07-01', '2025-07-01')
    test = [r for r in rows if '2025-07-01' <= r['date'] < '2026-07-01']
    if len(test) < 1000:
        raise ValueError('Incomplete 2025-26 chronological holdout')
    predictions = score(fitted, test)
    y = [r['y'] for r in test]
    evaluated = {name: metrics(y, p) for name, p in predictions.items()}
    beats_basic = all(evaluated['moneyline_calibrated'][m] < evaluated['goals_only'][m] for m in ['brier','log_loss'])
    return {'release_status': 'experimental; one-season validation only',
            'beats_goals_only_on_brier_and_log_loss': beats_basic,
            'promotion_warning': None if beats_basic else 'Richer ML model does not beat the goals/rest-only baseline. Do not treat feature complexity as evidence of better forecasts.',
            'protocol': {'train': 'before 2024-07-01', 'calibration': '2024-07-01 to 2025-06-30',
                         'test': '2025-07-01 to 2026-06-30', 'train_games': fitted['train_games'],
                         'calibration_games': fitted['calibration_games']},
            'metrics': evaluated,
            'limitations': ['Single untouched season holdout; no market-price or profitability test.',
                            'MoneyPuck historical xG is a current retrospective snapshot, not point-in-time archived xG.',
                            'Goalies, injuries and travel are not model inputs. Home advantage is learned by the intercept.']}, [
                {'id': r['id'], 'date': r['date'], 'actual_home_win': r['y'],
                 **{name: float(p[i]) for name, p in predictions.items()}} for i, r in enumerate(test)]
