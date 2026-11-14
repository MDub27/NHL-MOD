# NHL data and moneyline development

This branch preserves the exact V1 estimator and V2 model from commit
`d9ef50ec8ced61f6067b5aa084df312d04da864e`. It adds a separate data layer,
moneyline model and report. It is not merged or promoted automatically.

## Explicit execution only

After an approved merge, open **NHL 4.5 Development Manual** in Actions,
choose the desired branch, enter an explicit NHL date such as `2026-11-14`,
and choose `all`, `predict` or `backtest`. ChatGPT Work can perform these steps
using its signed-in cloud browser. No new credentials or external dispatch
service is needed. The original V2 launcher remains unchanged.

Before merge, integration tests used the registered V2 workflow path with a
development-only workflow definition at commit
`640c39fde884325754f807982e8b20bd9eba1faa`. The final PR moves this exact development
workflow to `.github/workflows/nhl-v3.yml` and restores the original V2 launcher.
The new workflow becomes independently available after an approved merge.

Local commands:

```sh
python -m pip install -r requirements-v3.txt
python -m unittest discover -s tests -p test_v3.py -v
python nhl_report.py --date 2026-11-14 --mode all
```

The branch removes V1's old cron trigger. This removal takes effect only after
an approved merge; until then main's existing V1 schedule remains active.

## Data safeguards

- NHL API is authoritative for schedules, final scores and win labels, including
  overtime/shootout winners. In-progress scores never enter history.
- Retrieve all 32 club schedules and cross-check each game appears on both teams'
  feeds with matching fields. Require published season counts (82 per club for
  2023–25; 84 for 2026). Unknown future season formats stop for an explicit update.
- Missing/invalid JSON, final scores, duplicate games, incomplete schedules,
  unresolved prior-date games, conflicting records and requested-date mismatches
  stop execution. Postponed prior games require reconciliation rather than silent omission.
- HTTP requests have a 20-second timeout and up to three attempts for transient
  failures (408/429/5xx or timeouts); permanent 4xx errors stop immediately.
- Closed seasons cache indefinitely after validation; current seasons refresh
  after one hour; requested-date schedules after five minutes. MoneyPuck compact
  xG refreshes daily. Invalid refreshes do not overwrite existing cache files.
- Cache is restored/saved in GitHub Actions. The compact xG cache avoids repeatedly
  downloading the large source CSV. A cold xG fetch still downloads it once.
- All training and rolling features use completed games with NHL `gameDate`
  strictly before the requested date. Same-day games are processed together in
  chronological tests. The target is never inferred from UTC today.
- Games whose scheduled starts have passed are marked reconstructions, excluded
  from ranked selections/combinations, and excluded from the pregame-only snapshot.

NHL published 2026 schedule reference:
https://www.nhl.com/news/nhl-announces-2026-27-regular-season-schedule

## Moneyline model and available inputs

The separate regularized logistic classifier estimates the probability of the
home team winning the **whole game, including overtime and shootouts**. Away
probability is its complement. Features are 50-game goals for/against with a
15-game prior at three goals per team, last-10 goal differential, 50-game xG
for/against, last-10 xG differential, rest-day difference and back-to-back
difference. The fitted intercept supplies average home advantage.

MoneyPuck.com provides the downloaded team xG data, filtered to `situation=all`
and regular-season game IDs. Credit: https://moneypuck.com/data.htm . Its data
terms describe non-commercial use; obtain permission for other uses. Missing xG
is explicit and imputed using training-only medians, with a missing-data flag.
MoneyPuck xG is a retrospective downloaded snapshot: this is date-safe by game,
but not a strict historical release-time data vintage. That limitation is included
in reports. SHA256 fingerprints and fetch times are recorded.

**Unavailable, not invented:** verified starting goalies, goalie performance
matched to archived pregame confirmations, timestamped injuries and reliable
travel distances/timezones. Reports mark goalies unconfirmed and these features
planned. The model does not claim goalie/injury-adjusted probabilities. Neutral
site effects are not separately modeled.

## Fixed chronological evaluation

The holdout protocol is fixed before inspecting results:

- 2023–24: train after 200-game feature warmup.
- 2024–25: separate Platt calibration.
- 2025–26: untouched season test; no tuning on it.

Report Brier score, log loss, accuracy, actual/estimated home win rates, calibration
bins and expected calibration error. Compare raw and calibrated full models to
a goals/rest-only classifier, a training-only home-win-rate constant, 50/50 and
a fixed Elo baseline (K=20, home bonus 50). Lower Brier/log loss is better.
Adding xG or calibration is not assumed to improve performance. One test season
is preliminary validation, not proof of market edge or profitability.

Live fitting uses earlier seasons for training and the most recent completed
season for calibration. For a 2026 target this means training before July 2025
and calibration in 2025–26; the live fit differs from the earlier holdout fit.

## Combined report and retention

Each timestamped directory contains `report.json`, `report.md`, `games.csv` and
`pregame.json`. It includes both teams, Pacific and UTC starts, original expected
total, V1 original and V2 calibrated Over 4.5, both ML probabilities, goalie
status, data warnings and provenance. Top five totals and ML candidates exclude
already-started games. No precise confidence interval is claimed.

Distinct-game two/three-leg combinations show a product **under an explicit
independence assumption**, which does not estimate common external risks. Same-game
selections are never multiplied: only Frechet bounds are supplied. This is not
a fitted joint outcome model. No market prices are supplied, so expected value
is null and no positive-EV claim is made.

Snapshots never overwrite an existing timestamp directory. Artifacts have unique
run/attempt IDs and GitHub retains them for **90 days**, not permanently. Download
or archive them before expiry when long-term retention is needed. Cache is only
an optimization and can be evicted. Predictions are never reconstructed and
presented as an original saved pregame forecast.

## Preserved V2 test failure

The original `test_probability_bounds_and_metrics` fails because V2 rounds
`0.999999` to `1.0`. V2 source and tests are preserved unchanged. Its separate
workflow regression step is explicitly `continue-on-error`, and its log is
included in artifacts; new data/moneyline tests are blocking. Do not describe
the entire suite as passing. Fixing V2 requires a separately approved version.

## Validation results

The cold-cache integration run succeeded:
https://github.com/MDub27/NHL-MOD/actions/runs/43627576706

It retrieved 4,247 completed games dated before 2026-11-14 and generated a
14-game report (13 pregame-eligible at generation). The artifact includes
all reports, backtest predictions, metrics and separate test logs. Artifact ZIP
SHA256: `23c8c3f16db6510bd784631820b9c3d9addc951f7123667cbe8133411876741e`.

Holdout: 1,312 games in 2025–26, using 1,109 training and 1,312 calibration games.

| Model | Brier | Log loss | Calibration error |
|---|---:|---:|---:|
| Full xG model, raw | 0.248724 | 0.690523 | 0.055629 |
| Full xG model, calibrated | 0.248674 | 0.690425 | 0.068567 |
| Goals/rest-only baseline | 0.247013 | 0.687185 | 0.039080 |
| Training home-rate baseline | 0.250015 | 0.693183 | 0.022451 |
| 50/50 baseline | 0.250000 | 0.693147 | 0.022104 |
| Fixed Elo baseline | 0.254193 | 0.703325 | 0.075472 |

The richer model does **not** beat the goals/rest-only baseline. Calibration
slightly improves Brier/log loss but worsens binned calibration error. It remains
an experimental research candidate; do not promote it as a superior betting model.
No parameters were retuned after inspecting this holdout.

Tests: 20 new tests and three preserved player-feature tests passed. The four
preserved V2 tests contain one known failure described above. The first integration
attempt caught the weekly NHL schedule's parent-day date representation; that
parser was fixed and a regression test added before the successful run.


Warm-cache verification also succeeded:
https://github.com/MDub27/NHL-MOD/actions/runs/43627754032
All 130 source-cache events were hits, with no source redownloads. Predictions
matched the cold run exactly and the timestamped directories differed. The
compressed cache was approximately 1.4 MB.

A copy of the successful pregame report and metrics is committed under
`validation/2026-11-14/` so this reviewed result survives artifact expiry.
