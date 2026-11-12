# NHL 4.5 V3 — experimental simulation branch

This is an **initial engine, not a verified predictive upgrade**. It does not automatically collect sportsbook odds, projected lineups, injuries, or confirmed goalies. It requires timestamped pregame inputs supplied by a separate data acquisition process. No scheduled runs are added.

## Goal

Given an observed sportsbook total L (half-goal), compare **Over (L − 2)** and **Under (L + 2)**, including the actual quoted price and fees. Simulate 1,000 scorelines with reproducible random seed and report total-goal distribution, central 90% interval, hit probabilities and estimated net returns.

## Inputs

Create `game.json` with actual verified game-day inputs, for example (numbers below are fictional and must not be treated as predictions):

```json
{
  "as_of": "2026-11-10",
  "source_timestamp_utc": "2026-11-10T18:00:00Z",
  "book_total": 6.5,
  "over_decimal": 1.25,
  "under_decimal": 1.15,
  "fee_fraction": 0,
  "simulations": 1000,
  "seed": 42,
  "home": {
    "name": "BOS",
    "team_expected_goals": 3.1,
    "baseline_player_goals_per_game": 0.5,
    "lineup": [{"id": 1, "position": "C", "goals_per_game": 0.5}],
    "opposing_goalie_effect": 0
  },
  "away": {
    "name": "NYR",
    "team_expected_goals": 2.7,
    "baseline_player_goals_per_game": 0.4,
    "lineup": [{"id": 2, "position": "C", "goals_per_game": 0.4}],
    "opposing_goalie_effect": 0
  }
}
```

Run: `python nhl_v3.py --input game.json --output output/v3_simulation.json`

Run tests: `python -m unittest discover -s tests -v`

## Model caveats

- Team expected-goal baselines must be calculated from **only pregame historical data**, e.g. V1. The input is not fetched automatically.
- A real, complete projected lineup is required; a single player in the example is a synthetic fixture only.
- The player grade and similarity function are exploratory. The coefficients are **not fitted** and are not yet linked to expected-goal adjustments.
- The player multiplier is a conservative, hand-chosen shrinkage heuristic. It must be fitted and validated before real-world use.
- The shared lognormal pace shock adds score correlation but its dispersion is not empirically fitted.
- The best/worst simulated totals are random sample extremes, not credible bounds.
- Quoted decimal odds and fees must reflect the **actual teased contracts**, not the original sportsbook total. Without both prices the engine reports hit rates, not value.
- No head-to-head V1/V2/V3 backtest is valid without archived, as-of-date lineup and market prices. Do not substitute today's lineups or odds into historical games.
- Calibrate against held-out dates, compare Brier/log loss and a constant-rate baseline, and do not merge until data integrity and code tests pass.

## Next engineering milestones

1. Build a timestamped data ingestion layer: player game logs, active roster, projected lines, confirmed goalie, injuries, game-day totals and prices; preserve raw source metadata.
2. Compute position/role-specific grades and comparable players from rolling as-of-date features; use prior seasons only as priors.
3. Fit player and goalie adjustments and pace variance using chronological training games; test against held-out seasons.
4. Create a locked pregame prediction archive and an after-final evaluation job; avoid hindsight leakage.
5. Compare V1/V2/V3 over identical historical fixtures and teaser prices. Run manually only.
