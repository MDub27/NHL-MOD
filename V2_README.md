# NHL 4.5 Version 2 (proposed)

Version 1 remains in `nhl_model.py` unchanged. Version 2 is `nhl_v2.py`.

## Manual runs only
GitHub Actions → **NHL 4.5 V2 Manual** → **Run workflow** → select date (YYYY-MM-DD) and mode.

- **predict**: all not-yet-final regular-season games for the selected date, with original and calibrated Over 4.5 probabilities. Run **before** puck drop and retain the downloadable artifact.
- **backtest**: fit Platt/logistic calibration on chronological V1 backtest predictions before 2025-10-01, validate from 2025-10-01 to 2026-07-01. Reports Brier and log loss for both.
- **evaluate**: reconstruct the selected completed date using strictly earlier scores and compare outcomes. This is not an audit of a previously saved live prediction.

Calibration uses historical *out-of-sample-by-day* predictions, not the same games used as estimator inputs. The train/validation split is chronological. The model is not considered improved until held-out Brier/log loss actually improve.

The V1 workflow has its automatic schedule removed. No scheduled run is configured for V2.

Caveats: NHL schedule data may be revised; cache is not immutable; the existing V1 estimator uses historical season schedules. For rigorous live evaluation, preserve pregame artifacts and compare to final scores separately. This project is experimental; percentages are not guaranteed or validated as profitable.

## Offline comparison from saved historical CSV (not a GitHub Actions run)

The uploaded `backtest_predictions.csv` contains 1,596 records dated 2025-10-07 through 2026-11-09. An independent reproduction of the Platt-scaling equations used 1,312 rows dated before 2026-09-01 for calibration and 284 later rows as a holdout:

| Method | Holdout Brier | Holdout log loss | Mean predicted Over 4.5 |
|---|---:|---:|---:|
| Original V1 | 0.153895 | 0.488819 | 0.74199 |
| Calibrated V2 | 0.147610 | 0.471528 | 0.80923 |
| Constant training-set Over 4.5 rate | 0.147526 | 0.471394 | 0.80640 |

Holdout actual Over 4.5 rate: 0.82042. **The constant baseline marginally beats V2**, so predictive value beyond the base rate is not established. This is an exploratory alternate split, not the V2 CLI's default 2024/2025 chronological split. The saved CSV's game outcomes have not been independently reconciled against the NHL API. Do not treat these numbers as proof of production accuracy.

## Tests
Run `python -m unittest discover -s tests -v` to check calibration bounds, scoring, and holdout leakage safeguards. A full NHL API-backed integration test and GitHub Actions run remain outstanding; keep the PR as a draft until those succeed.
