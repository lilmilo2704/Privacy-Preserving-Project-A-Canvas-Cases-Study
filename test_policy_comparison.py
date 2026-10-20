"""Independent checks for coverage constraints and saved paired releases."""
import json
import unittest
from collections import Counter
import numpy as np
import pandas as pd
from experiment import ROOT,QI,PUBLIC,read_config,transform,sha256
from compare_policies import EXT,programme_suppression,policy_candidate,select_policy_release


class PolicyChecks(unittest.TestCase):
    def setUp(self):self.cfg=read_config()

    def fixture(self,rare):
        rows=[]
        for p,n in [("P01",100),("P06",40)]:
            for i in range(n):
                unusual=p=="P06" and i<rare
                rows.append(dict(student_id=f"{p}_{i}",age=30+i if unusual else 20,
                                 postcode="9100",programme=p,mark=60,result_band="Pass"))
        return pd.DataFrame(rows)

    def test_programme_cap_accepts_exactly_ten_percent(self):
        raw=self.fixture(4)
        _,keep,info=policy_candidate(raw,self.cfg,0,0,5,1)
        self.assertTrue(info["feasible"])
        self.assertEqual(int((~keep).sum()),4)
        self.assertEqual(info["suppression_P06"],.10)

    def test_overall_cap_can_pass_while_programme_cap_fails(self):
        raw=self.fixture(5)
        _,_,info=policy_candidate(raw,self.cfg,0,0,5,1)
        self.assertTrue(info["overall_feasible"])
        self.assertFalse(info["programme_cap_feasible"])
        self.assertFalse(info["feasible"])

    def test_zero_suppression_exact_release_does_not_need_extra_generalisation(self):
        raw=self.fixture(0)
        _,_,chosen,_=select_policy_release(raw,self.cfg,5,1)
        self.assertEqual(chosen["age_level"],0)
        self.assertEqual(chosen["postcode_level"],0)
        self.assertEqual(chosen["generalisation_loss"],0)

    def test_single_band_programme_remains_infeasible_for_l_two(self):
        release,_,_,trace=select_policy_release(self.fixture(0),self.cfg,5,2)
        self.assertIsNone(release)
        self.assertFalse(trace.feasible.any())

    def test_original_inputs_and_outputs_are_byte_identical(self):
        lock=json.loads((EXT/"design_lock.json").read_text())
        for path,digest in lock["original_files_before"].items():
            self.assertEqual(sha256(ROOT/path),digest,path)
        self.assertEqual(sha256(EXT/"config.json"),lock["extension_config_sha256"])
        self.assertEqual(sha256(ROOT/"compare_policies.py"),lock["extension_script_sha256"])

    def test_all_paired_settings_present_without_new_students(self):
        paired=pd.read_csv(EXT/"results/paired_deltas.csv")
        self.assertEqual(len(paired),45)
        self.assertEqual(paired[["seed","method","k","ell"]].drop_duplicates().shape[0],45)
        self.assertTrue(paired.feasible_programme_cap.all())
        self.assertTrue((paired.delta_generalisation_loss>=-1e-12).all())
        self.assertTrue((paired.max_programme_suppression_programme_cap<=.1+1e-12).all())

    def test_saved_release_constraints_and_multisets(self):
        metrics=pd.read_csv(EXT/"results/new_policy_metrics.csv")
        self.assertEqual(len(metrics),45)
        self.assertEqual(len(list((EXT/"releases").glob("*.csv"))),45)
        for _,setting in metrics.iterrows():
            seed,k,ell=int(setting.seed),int(setting.k),int(setting.ell)
            stem=f"seed{seed}_{setting.method}_k{k}"
            raw=pd.read_csv(ROOT/f"private/students_seed{seed}.csv",dtype={"postcode":str})
            membership=pd.read_csv(EXT/f"audit/{stem}_membership.csv")
            self.assertEqual(membership.student_id.tolist(),raw.student_id.tolist())
            keep=membership.retained.astype(bool)
            by_programme=programme_suppression(raw,keep)
            self.assertTrue((by_programme.n_removed<=.10*by_programme.n_original+1e-9).all())
            release=pd.read_csv(EXT/f"releases/{stem}.csv",dtype={"age":str,"postcode":str})
            self.assertEqual(release.columns.tolist(),PUBLIC)
            self.assertEqual(len(release),int(keep.sum()))
            mapped,_=transform(raw,self.cfg,int(setting.age_level),int(setting.postcode_level))
            expected=Counter(map(tuple,mapped.loc[keep,PUBLIC].to_numpy()))
            self.assertEqual(Counter(map(tuple,release[PUBLIC].to_numpy())),expected)
            group=release.groupby(QI).result_band.agg(["size","nunique"])
            self.assertGreaterEqual(group["size"].min(),k)
            self.assertGreaterEqual(group["nunique"].min(),ell)

    def test_programme_metrics_use_full_original_denominator(self):
        table=pd.read_csv(EXT/"results/combined_programme_metrics.csv")
        new=table[table.policy=="overall_and_programme"]
        self.assertEqual(len(new),270)
        self.assertTrue((new.n_original==new.programme.map(self.cfg["programme_counts"])).all())
        self.assertTrue((new.n_released>=.90*new.n_original-1e-9).all())
        self.assertTrue(np.allclose(new.absolute_error_pp,100*(new.released_below50_rate-new.original_below50_rate).abs()))


if __name__=="__main__":unittest.main(verbosity=2)
