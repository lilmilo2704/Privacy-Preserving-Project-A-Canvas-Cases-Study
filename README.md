# Assignment 2 - supporting evidence

The report is the primary submission. This supplement provides the evidence behind its figures and numerical claims. All records, programme labels and postcode tokens are synthetic; UTS supplies grading boundaries only.

## Suggested reading order

1. Read the report. Its two figures are also supplied in `report/assets/`.
2. Consult the summary below and `policy_comparison/results/summary_mean_min_max.csv` for the main comparison.
3. For the small example, open `worked_example/private_original.csv`, then the four `*_release.csv` files and `metrics.csv`. The four `*_groups.csv` files show every equivalence class, allowing the dataset-wide claims to be checked.
4. Read `source_register.json`, `data_dictionary.csv` and `config.json` for origins and generation assumptions. `record_provenance.csv` records all 5,020 source records.
5. Code, tests and additional evidence are available below if detailed verification is needed. No code execution is needed to inspect the saved results.

## Main comparison used in the report

Values are five-seed means for distinct l=3. MAE is the equally weighted mean error across six programme below-50 rates. Generalisation loss measures demographic detail lost. Full ranges are in the summary CSV.

| k | Suppression policy | MAE (percentage points) | Overall suppression (%) | Generalisation loss |
| ---: | --- | ---: | ---: | ---: |
| 5 | Overall | 1.353 | 8.34 | 0.211 |
| 5 | Overall + programme | 0.485 | 1.26 | 0.496 |
| 10 | Overall | 1.032 | 8.46 | 0.317 |
| 10 | Overall + programme | 0.508 | 1.28 | 0.702 |
| 20 | Overall | 1.387 | 7.80 | 0.625 |
| 20 | Overall + programme | 0.313 | 1.84 | 0.881 |

The report recommends k=5,l=3 with both caps. This preserves more demographic detail than the larger tested k values while satisfying the declared privacy and coverage conditions. It is a project policy choice within this simulation, not a universal optimum.

## Evidence map

| Report claim | Evidence |
| --- | --- |
| Identifier-removal baseline and original defence results | `results/run_metrics.csv`; `results/programme_metrics.csv` |
| Both policies, all seeds, methods and k settings | `policy_comparison/results/combined_run_metrics.csv`; `combined_programme_metrics.csv` in the same folder |
| Five-seed means and full ranges | `policy_comparison/results/summary_mean_min_max.csv` |
| 42 improvements and three increases | `policy_comparison/results/paired_deltas.csv`; the three increases share one seed/k outcome |
| Complete 20-student example | `worked_example/` |
| Original synthetic source records | Five `private/students_seed*.csv` files |
| 57.5% P06 suppression example | `releases/seed2026_k+l3_k10.csv`; matching membership in `audit/`; source seed 2026 and programme metrics |
| Representative recommended release | `policy_comparison/releases/seed2026_k+l3_k5.csv`; matching membership and groups in `policy_comparison/audit/` |
| 22 recorded verification checks | `policy_comparison/results/verification_log.txt`; source tests in `test_experiment.py` and `test_policy_comparison.py` |
| Original 270-file preservation check | `policy_comparison/results/original_preservation_check.json`; original digest list in `policy_comparison/design_lock.json` |
| Sources, value definitions and record origins | `source_register.json`; `data_dictionary.csv`; `record_provenance.csv` |
| Current supplied-file integrity | `evidence_manifest.csv` lists included files and their SHA-256 hashes |

## Worked utility calculation

For P06, seed 2026, k=10,l=3 under the overall-only policy, the source has 21/40 below-50 results and the release has 9/17. Its absolute rate error is 100 x |9/17 - 21/40| = 0.441176 percentage points. Average all six programme errors for that run to obtain MAE = 0.735369 percentage points. This is one run, not the five-seed mean.

## Scope of this focused package

Historical review notes, report-building scripts, alternative unused figures and hundreds of per-target audits and release files have been omitted. All baseline and paired run metrics, programme metrics and the complete small example are retained. The retained representative releases are labelled by seed and setting; they are not the complete experiment.

The 270-file preservation result and 22-test log document the original complete execution. Many files listed in the historical design lock are intentionally absent from this compact archive. They can be regenerated from the unchanged code and configurations. This compact archive has its own `evidence_manifest.csv`; it does not claim to include all 270 original files. The original full review archive remains a separate project record and need not be submitted alongside this one.

Source records and evaluation membership are supplied only because this is synthetic data. The evidence package is not a proposal to publish private mappings or multiple alternative releases of real students.

## Reproduction (optional)

Use a fresh extracted copy. The recorded environment is in `environment.json`, with pinned dependencies in `requirements.txt`. Run these commands from the extracted `assignment2_anonymisation` folder:

```text
python -m pip install -r requirements.txt
python experiment.py
python compare_policies.py
python -m unittest -v
```

Run both experiment scripts before the test suite: the tests inspect the full set of generated releases and audits. Regeneration writes all omitted experimental outputs and updates generated files. The packaged historical verification log is retained evidence of the earlier execution; new test results appear in the terminal. This package contains no claim that the student personally reran the experiment or captured execution screenshots.
