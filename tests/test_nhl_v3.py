import unittest
from nhl_v3 import simulate,evaluate_market,comparable_players,analyze

class SimulationTests(unittest.TestCase):
    def test_seed_is_reproducible(self):
        self.assertEqual(simulate(3,2,1000,42),simulate(3,2,1000,42))
    def test_two_goal_tease(self):
        games=[{"total":t} for t in [4,5,6,7,8,9]]
        choices=evaluate_market(games,6.5,1.2,1.1)
        self.assertEqual(choices[0]["teased_line"],4.5)
        self.assertEqual(choices[1]["teased_line"],8.5)
        self.assertAlmostEqual(choices[0]["probability"],5/6,places=4)
        self.assertAlmostEqual(choices[1]["probability"],5/6,places=4)
    def test_underdog_price_matters(self):
        games=[{"total":6}]*8+[{"total":10}]*2
        choices=evaluate_market(games,6.5,1.5,1.05)
        self.assertGreater(choices[0]["expected_net_return_per_unit"],
                           choices[1]["expected_net_return_per_unit"])
    def test_comparables_match_position(self):
        target={"id":1,"position":"C","goals_per_game":.4}
        pool=[{"id":2,"position":"C","goals_per_game":.42},
              {"id":3,"position":"D","goals_per_game":.4}]
        self.assertEqual(comparable_players(target,pool)[0]["id"],2)
        self.assertEqual(len(comparable_players(target,pool)),1)
    def test_requires_market_timestamp(self):
        with self.assertRaises(ValueError):
            analyze({"as_of":"2026-11-10","book_total":6.5})
    def test_complete_simulation(self):
        lineup=[{"id":1,"goals_per_game":.3}]
        payload={"as_of":"2026-11-10","source_timestamp_utc":"2026-11-10T18:00:00Z",
          "book_total":6.5,"over_decimal":1.3,"under_decimal":1.2,
          "home":{"name":"BOS","lineup":lineup,"baseline_player_goals_per_game":.3,"team_expected_goals":3},
          "away":{"name":"NYR","lineup":lineup,"baseline_player_goals_per_game":.3,"team_expected_goals":2.8}}
        result=analyze(payload)
        self.assertEqual(result["simulation"]["count"],1000)
        self.assertEqual(len(result["teased_markets"]),2)

if __name__=="__main__":
    unittest.main()
