"""Run: python -m unittest discover -s tests -v"""
import unittest
from nhl_player_features import player_features, lineup_scoring_features

class PlayerFeatureTests(unittest.TestCase):
    def fetch(self, path):
        season = int(path.split("/")[-2][:4])
        # Include a future game that must never affect an as-of-date prediction.
        rows = ([{"gameDate":"2024-11-01","goals":1,"shots":4,"points":2,"powerPlayPoints":1}] * 10
                if season == 2024 else
                [{"gameDate":"2026-11-08","goals":0,"shots":2,"points":1,"powerPlayPoints":0},
                 {"gameDate":"2026-11-10","goals":10,"shots":20,"points":15,"powerPlayPoints":5}])
        return {"gameLog":rows}

    def test_as_of_excludes_same_day(self):
        p=player_features(123,"2026-11-10",lookback_seasons=2,fetch=self.fetch)
        self.assertEqual(p["current_season_games"],1)
        self.assertEqual(p["goals_per_game"]["current_season"],0)
        self.assertEqual(p["goals_per_game"]["prior_season_median"],1)

    def test_lineup_requires_explicit_players(self):
        with self.assertRaises(ValueError):
            lineup_scoring_features([],"2026-11-10",fetch=self.fetch)

    def test_lineup_aggregate(self):
        r=lineup_scoring_features([123,456],"2026-11-10",fetch=self.fetch)
        self.assertEqual(r["player_count"],2)
        self.assertGreater(r["sum_player_goal_rates"],0)

if __name__=="__main__":
    unittest.main()
