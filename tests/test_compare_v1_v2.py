import unittest
from compare_v1_v2 import evaluate

class WalkForwardTests(unittest.TestCase):
    def test_no_same_day_training_and_monotonic_ranking(self):
        rows=[]
        for day in range(1,7):
            for i,p in enumerate((.6,.7,.8,.9)):
                rows.append({"date":f"2026-01-{day:02d}","home":f"H{i}","away":f"A{i}",
                             "probability_over_4_5":p,"actual":int((day+i)%3!=0)})
        report=evaluate(rows,min_train=12,top_k=3)
        self.assertEqual(report["days"],3)
        self.assertEqual(report["games"],12)
        self.assertEqual(report["top_k_changed_days"],0)
    def test_invalid_outcome_rejected(self):
        with self.assertRaises(ValueError):
            evaluate([{"date":"2026-01-01","home":"A","away":"B",
                       "probability_over_4_5":.8,"actual":2}],min_train=0)

if __name__=="__main__":unittest.main()
