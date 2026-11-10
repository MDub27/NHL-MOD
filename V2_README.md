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
