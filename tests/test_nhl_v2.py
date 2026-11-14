"""Offline tests: run python -m unittest discover -s tests -v."""
import unittest
from nhl_v2 import fit_calibrator, calibrated, metrics, split_and_fit

class CalibrationTests(unittest.TestCase):
    def sample(self, date, p, actual):
        return {"date":date,"home":"BOS","away":"NYR","probability_over_4_5":p,"actual":actual}

    def test_identity_with_insufficient_training(self):
        params=fit_calibrator([self.sample("2024-01-01",.7,1)])
        self.assertEqual(params["status"],"insufficient_data")
        self.assertAlmostEqual(calibrated(.7,params),.7,places=5)

    def test_chronological_holdout_not_used_for_fitting(self):
        train=[self.sample("2024-10-01",.72,i%2) for i in range(120)]
        test=[self.sample("2025-10-01",.72,1) for _ in range(25)]
        p1,scored1=split_and_fit(train+test,"2025-10-01","2026-07-01")
        changed=[{**r,"actual":0} for r in test]
        p2,scored2=split_and_fit(train+changed,"2025-10-01","2026-07-01")
        self.assertEqual(p1,p2)
        self.assertEqual([r["calibrated_probability"] for r in scored1],
                         [r["calibrated_probability"] for r in scored2])

    def test_probability_bounds_and_metrics(self):
        rows=[self.sample("2025-10-01",.9,1),self.sample("2025-10-02",.1,0)]
        self.assertGreater(calibrated(.999999,{"intercept":0,"slope":1}),0)
        self.assertLess(calibrated(.999999,{"intercept":0,"slope":1}),1)
        m=metrics(rows,"probability_over_4_5")
        self.assertAlmostEqual(m["brier"],.01)
        self.assertEqual(m["games"],2)

    def test_empty_validation(self):
        params,rows=split_and_fit([self.sample("2024-10-01",.8,1)],
                                  "2025-10-01","2026-07-01")
        self.assertEqual(rows,[])
        self.assertEqual(metrics(rows,"probability_over_4_5")["games"],0)

if __name__=="__main__":
    unittest.main()
