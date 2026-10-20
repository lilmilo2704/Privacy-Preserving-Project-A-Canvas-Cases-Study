"""Checks attack semantics, failure cases, release constraints and reproducibility."""
import unittest
import numpy as np
import pandas as pd
from experiment import (ROOT, QI, PUBLIC, read_config, generate, transform, candidate,
                        select_release, attack, evaluate, worked_example, validate_release)


class ExperimentChecks(unittest.TestCase):
    def setUp(self):
        self.cfg=read_config()

    def fixture(self,bands):
        return pd.DataFrame([dict(student_id=f"T{i}",age=20,postcode="9100",programme="P01",mark=40 if b=="Below50" else 60,result_band=b) for i,b in enumerate(bands)])

    def test_seed_reproducibility_and_changed_seed(self):
        first=generate(self.cfg,2026)
        pd.testing.assert_frame_equal(first,generate(self.cfg,2026))
        self.assertFalse(first.equals(generate(self.cfg,2027)))

    def test_generated_domain_and_mark_band_consistency(self):
        raw=generate(self.cfg,2026)
        self.assertEqual(len(raw),1000)
        self.assertEqual(raw.student_id.nunique(),1000)
        self.assertEqual(raw.programme.value_counts().to_dict(),self.cfg["programme_counts"])
        self.assertTrue(raw.age.between(18,65).all())
        for band,(lo,hi) in zip(self.cfg["band_order"],self.cfg["mark_intervals_inclusive"]):
            self.assertTrue(raw.loc[raw.result_band==band,"mark"].between(lo,hi).all())

    def test_hierarchy_contains_every_true_age_and_postcode(self):
        raw=generate(self.cfg,2026)
        for a in range(5):
            for p in range(4):
                release,loss=transform(raw,self.cfg,a,p)
                self.assertTrue(0 <= loss <= 1)
                for i in raw.index:
                    parts=release.loc[i,"age"].split("-")
                    self.assertTrue(int(parts[0]) <= raw.loc[i,"age"] <= int(parts[-1]))
                    self.assertTrue(raw.loc[i,"postcode"].startswith(release.loc[i,"postcode"].rstrip("*")))

    def test_baseline_unique_attack_in_worked_example(self):
        raw=worked_example()
        release,_=transform(raw,self.cfg,0,0)
        for _,target in raw.iterrows():
            result=attack(target[QI].to_dict(),release,self.cfg)
            self.assertEqual(result["candidates"],1)
            self.assertEqual(result["modal_band"],target.result_band)

    def test_k_only_can_leave_complete_sensitive_disclosure(self):
        raw=worked_example()
        release,keep,_,_=select_release(raw,self.cfg,5,1)
        validate_release(raw,release,keep,5,1,self.cfg)
        metrics,_,_=evaluate(raw,release,keep,self.cfg)
        self.assertEqual(metrics["unique_link_pct"],0)
        self.assertEqual(metrics["homogeneous_band_pct"],100)

    def test_l_diversity_in_worked_example_removes_single_band_groups(self):
        raw=worked_example()
        for ell in [2,3]:
            release,keep,_,_=select_release(raw,self.cfg,5,ell)
            validate_release(raw,release,keep,5,ell,self.cfg)
            metrics,_,_=evaluate(raw,release,keep,self.cfg)
            self.assertEqual(metrics["homogeneous_band_pct"],0)
            self.assertEqual(metrics["suppression"],0)

    def test_distinct_l_does_not_bound_confidence_by_one_over_l(self):
        raw=self.fixture(["Pass"]*19+["Below50"])
        release,keep,_,_=select_release(raw,self.cfg,5,2)
        result=attack(raw.iloc[0][QI].to_dict(),release,self.cfg)
        self.assertEqual(result["band_confidence"],.95)
        self.assertGreater(result["band_confidence"],1/2)

    def test_l_diversity_can_still_reveal_binary_status(self):
        raw=self.fixture(["Pass"]*10+["Credit"]*10)
        release,keep,_,_=select_release(raw,self.cfg,5,2)
        metrics,_,_=evaluate(raw,release,keep,self.cfg)
        self.assertEqual(metrics["homogeneous_band_pct"],0)
        self.assertEqual(metrics["certain_binary_status_pct"],100)

    def test_infeasible_single_band_programme(self):
        raw=self.fixture(["Pass"]*20)
        release,keep,chosen,trace=select_release(raw,self.cfg,5,2)
        self.assertIsNone(release)
        self.assertFalse(trace.feasible.any())

    def test_ten_percent_suppression_boundary(self):
        for rare,expected in [(10,True),(11,False)]:
            raw=self.fixture(["Pass"]*100)
            raw.loc[:rare-1,"programme"]="P06"
            raw.loc[:rare-1,"age"]=np.arange(30,30+rare)
            _,_,info=candidate(raw,self.cfg,0,0,5,1)
            self.assertEqual(info["feasible"],expected)

    def test_unknown_target_is_not_reported_as_success(self):
        raw=self.fixture(["Pass"]*20)
        release,_=transform(raw,self.cfg,0,0)
        result=attack(dict(age=20,postcode="9100",programme="P99"),release,self.cfg)
        self.assertEqual(result["candidates"],0)
        self.assertTrue(np.isnan(result["band_confidence"]))

    def test_row_order_does_not_change_attack(self):
        raw=worked_example()
        release,_,_,_=select_release(raw,self.cfg,5,2)
        target=raw.iloc[0][QI].to_dict()
        self.assertEqual(attack(target,release,self.cfg),attack(target,release.sample(frac=1,random_state=99),self.cfg))

    def test_selector_is_minimum_loss_within_declared_lattice(self):
        raw=generate(self.cfg,2026)
        _,_,chosen,trace=select_release(raw,self.cfg,10,3)
        expected=trace.loc[trace.feasible].sort_values(["generalisation_loss","suppression","age_level","postcode_level"]).iloc[0]
        self.assertEqual(chosen["age_level"],expected.age_level)
        self.assertEqual(chosen["postcode_level"],expected.postcode_level)

    def test_saved_releases_have_no_identifiers_and_keep_exact_programme(self):
        files=list((ROOT/"releases").glob("*.csv"))
        self.assertEqual(len(files),50)
        for path in files:
            release=pd.read_csv(path,dtype={"age":str,"postcode":str})
            self.assertEqual(list(release.columns),PUBLIC)
            self.assertTrue(set(release.programme) <= set(self.cfg["programme_counts"]))
            if "baseline" not in path.name:
                k=int(path.stem.rsplit("_k",1)[1])
                ell=3 if "k+l3" in path.name else 2 if "k+l2" in path.name else 1
                groups=release.groupby(QI).result_band.agg(["size","nunique"])
                self.assertGreaterEqual(groups["size"].min(),k)
                self.assertGreaterEqual(groups["nunique"].min(),ell)
                self.assertGreaterEqual(len(release),900)


if __name__=="__main__":
    unittest.main(verbosity=2)
