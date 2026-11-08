"""Hand-calculated raw-evidence checks; this module never loads model code."""
import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("pilot_analysis", ROOT / "scripts/analyze_stair_improvement.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class OfflineAnalysisTests(unittest.TestCase):
    def test_shortlist_uses_absolute_quality_and_collapses_identical_weights(self):
        def cell(name,method,r1,sha,seconds=1.,cost=10.):
            return {"cell":name,"mode":"constrained","method":method,"r1":r1,"checkpoint_sha256":sha,
                "pool":935,"budget":384,"selected_update":192,"query_seconds":seconds,"charged_training_seconds":cost}
        metrics=[cell("D","D",.20,"weak"),cell("E","E",.44,"flat",.5),
                 cell("F_expensive","F",.445,"hierarchy",1.,30.),cell("F_cheap","F",.445,"hierarchy",1.,10.)]
        effects=[{"lead":"quality lead","effect":"more_data_384","right":r["cell"],"right_mode":"constrained"}
                 for r in metrics]
        chosen=m.choose_shortlist(metrics,effects)
        self.assertEqual([r["cell"] for r in chosen],["E","F_cheap"])

    def test_hand_calculated_rank_and_invalid_prediction(self):
        queries = [{"query_id":"q", "gold_title":"Gold"}]
        predictions = [{"query_id":"q", "hypotheses":[
            {"section_id":None,"validity":"invalid"},
            {"section_id":"finance:000","validity":"valid"}]}]
        actual = m.summary(m.score(predictions,queries,[{"title":"Gold"}]))
        self.assertEqual(actual["r1"],0)
        self.assertEqual(actual["r3"],1)
        self.assertEqual(actual["invalid1"],1)
        self.assertAlmostEqual(actual["ndcg3"],1/m.math.log2(3))

    def test_duplicate_query_cannot_replace_a_missing_query(self):
        queries = [{"query_id":q, "gold_title":"Gold"} for q in ["a","b"]]
        predictions = [{"query_id":"a", "hypotheses":[]}]*2
        with self.assertRaises(ValueError):
            m.score(predictions,queries,[{"title":"Gold"}])

    def test_batch_elapsed_is_charged_once_with_short_last_batch(self):
        records = [{"timing":{"end_to_end_seconds":4.,"batch_size":4}} for _ in range(4)]
        records += [{"timing":{"end_to_end_seconds":2.,"batch_size":2}} for _ in range(2)]
        self.assertEqual(m.query_seconds(records),1.)

    def test_document_cluster_keeps_its_queries_together(self):
        paired = m.clustered([1,0], ["same-document","same-document"])
        self.assertEqual(paired["difference_pp"],50.)
        self.assertEqual(paired["ci95_pp"],[50.,50.])
        self.assertTrue(paired["unstable_small_support"])

    def test_wide_harm_benefit_interval_overrides_point_gain(self):
        self.assertEqual(m.lead_label({"difference_pp":8.,"ci95_pp":[-6.,15.]}),"inconclusive")
        self.assertEqual(m.lead_label({"difference_pp":6.,"ci95_pp":[-1.,10.]}),"quality lead")
        self.assertEqual(m.lead_label({"difference_pp":-.8,"ci95_pp":[-2.,1.]},.70,True),
                         "projected efficiency lead")


if __name__ == "__main__":
    unittest.main()
