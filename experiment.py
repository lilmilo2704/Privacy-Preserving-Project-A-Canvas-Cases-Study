"""Synthetic educational-data release experiment. Run: python experiment.py.

The private/ and audit/ directories are investigator-only evaluation inputs.
Only releases/ contains example microdata releases. Never distribute all
alternative releases about the same people: intersection attacks are not tested.
"""
from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
QI = ["age", "postcode", "programme"]
PUBLIC = QI + ["result_band"]


def read_config():
    return json.loads((ROOT / "config.json").read_text())


def postcode_domain(cfg):
    return [f"{region:02d}{sub}{leaf}"
            for region in cfg["postcode_regions"]
            for sub in range(cfg["postcode_subregions_per_region"])
            for leaf in range(cfg["postcode_leaves_per_subregion"])]


def generate(cfg, seed):
    """Fixed cohort counts; all probabilities are assumptions, not empirical estimates."""
    rng = np.random.default_rng(seed)
    programmes = list(cfg["programme_counts"])
    regions = cfg["postcode_regions"]
    rows = []
    assert sum(cfg["programme_counts"].values()) == cfg["n_students"]
    for p_index, (programme, count) in enumerate(cfg["programme_counts"].items()):
        preferred = regions[p_index % len(regions)]
        for _ in range(count):
            young = rng.random() < cfg["age_young_probability"]
            mu, sd = cfg["age_young_mean_sd" if young else "age_mature_mean_sd"]
            age = int(np.clip(np.rint(rng.normal(mu, sd)), *cfg["age_domain"]))
            context = int(rng.integers(0, 2))
            region = preferred if rng.random() < cfg["programme_preferred_region_probability"] else int(rng.choice([r for r in regions if r != preferred]))
            sub = context if rng.random() < cfg["postcode_subregion_study_context_probability"] else 1-context
            postcode = f"{region:02d}{sub}{rng.integers(0, cfg['postcode_leaves_per_subregion'])}"
            mix = cfg["study_context_mixture_weight"]
            probs = (1-mix) * np.array(cfg["programme_band_probabilities"][programme]) + mix * np.array(cfg["study_context_band_probabilities"][context])
            band_index = int(rng.choice(len(probs), p=probs))
            lo, hi = cfg["mark_intervals_inclusive"][band_index]
            rows.append({"student_id": f"S{len(rows)+1:04d}", "age": age,
                         "postcode": postcode, "programme": programme,
                         "mark": int(rng.integers(lo, hi+1)),
                         "result_band": cfg["band_order"][band_index]})
    return pd.DataFrame(rows).sample(frac=1, random_state=seed).reset_index(drop=True)


def transform(raw, cfg, age_level, postcode_level):
    """Full-domain generalisation with nested, public hierarchies."""
    mapped = raw[PUBLIC].copy()
    amin, amax = cfg["age_domain"]
    width = cfg["age_widths"][age_level]
    lo = amin + ((raw.age-amin)//width)*width
    hi = np.minimum(lo+width-1, amax)
    mapped["age"] = raw.age.astype(str) if width == 1 else lo.astype(str) + "-" + hi.astype(str)
    prefix = cfg["postcode_prefix_lengths"][postcode_level]
    mapped["postcode"] = raw.postcode.str[:prefix] + "*"*(4-prefix)
    domain = postcode_domain(cfg)
    counts = pd.Series(domain).str[:prefix].value_counts()
    postcode_loss = (raw.postcode.str[:prefix].map(counts)-1)/(len(domain)-1)
    age_loss = (hi-lo)/(amax-amin)
    # Programme is exact; loss averages only the two transformed QI fields.
    full_source_loss = float(((age_loss+postcode_loss)/2).mean())
    return mapped, full_source_loss


def candidate(raw, cfg, age_level, postcode_level, k, ell):
    mapped, loss = transform(raw, cfg, age_level, postcode_level)
    group = mapped.groupby(QI, sort=False)["result_band"]
    sizes = group.transform("size")
    diversity = group.transform("nunique")
    keep = (sizes >= k) & (diversity >= ell)
    removed = int((~keep).sum())
    suppression = removed/len(raw)
    feasible = bool(keep.any() and removed <= cfg["max_suppression"]*len(raw) + 1e-9)
    trace = dict(age_level=age_level, postcode_level=postcode_level,
                 generalisation_loss=loss, suppression=suppression,
                 retained=int(keep.sum()), feasible=feasible)
    return mapped.loc[keep].copy(), keep, trace


def select_release(raw, cfg, k, ell):
    choices = []
    traces = []
    for a in range(len(cfg["age_widths"])):
        for p in range(len(cfg["postcode_prefix_lengths"])):
            release, keep, trace = candidate(raw, cfg, a, p, k, ell)
            traces.append(trace)
            if trace["feasible"]:
                key = (trace["generalisation_loss"], trace["suppression"], a, p)
                choices.append((key, release, keep, trace))
    if not choices:
        return None, None, None, pd.DataFrame(traces)
    _, release, keep, chosen = min(choices, key=lambda x: x[0])
    return release, keep, chosen, pd.DataFrame(traces)


def matches(query, release):
    """The attacker sees a single target's QIs and public release, never private IDs."""
    def age_match(label):
        parts = label.split("-")
        return int(parts[0]) <= int(query["age"]) <= int(parts[-1])
    age = release.age.map(age_match)
    prefix = release.postcode.str.rstrip("*")
    postal = prefix.map(lambda x: str(query["postcode"]).startswith(x))
    return release.loc[age & postal & (release.programme == query["programme"])]


def attack(query, release, cfg):
    found = matches(query, release)
    if found.empty:
        return {"candidates": 0, "unique": False, "modal_band": None,
                "band_confidence": np.nan, "below50_probability": np.nan}
    counts = found.result_band.value_counts().reindex(cfg["band_order"], fill_value=0)
    mode = counts.idxmax()  # Public fixed tie-breaking order, not the true value.
    return {"candidates": len(found), "unique": len(found) == 1,
            "modal_band": mode, "band_confidence": float(counts.max()/len(found)),
            "below50_probability": float(counts["Below50"]/len(found))}


def evaluate(raw, release, keep, cfg):
    """All attack percentages condition on retained targets with known release membership."""
    targets = raw.loc[keep]
    by_qi = release.groupby(QI, sort=False)
    sizes = by_qi.size()
    band_counts = by_qi.result_band.value_counts().unstack(fill_value=0).reindex(columns=cfg["band_order"], fill_value=0)
    confidence = band_counts.max(axis=1)/sizes
    distinct = (band_counts > 0).sum(axis=1)
    fail_probability = band_counts["Below50"]/sizes
    n = len(release)
    weighted = lambda x: float((x*sizes).sum()/n)
    programme_prior = release.groupby("programme").result_band.value_counts().unstack(fill_value=0)
    prior_accuracy = float(programme_prior.max(axis=1).sum()/n)
    metrics = dict(n_released=n, suppression=1-n/len(raw), n_groups=len(sizes),
                   min_group_size=int(sizes.min()), min_distinct=int(distinct.min()),
                   unique_link_pct=100*weighted(sizes == 1),
                   uniform_link_probability=weighted(1/sizes),
                   homogeneous_band_pct=100*weighted(distinct == 1),
                   modal_band_accuracy_pct=100*weighted(confidence),
                   programme_only_accuracy_pct=100*prior_accuracy,
                   band_accuracy_gain_pp=100*(weighted(confidence)-prior_accuracy),
                   high_confidence_band_pct=100*weighted(confidence >= cfg["high_confidence_threshold"]-1e-12),
                   certain_below50_pct=100*weighted(fail_probability == 1),
                   certain_binary_status_pct=100*weighted((fail_probability == 0) | (fail_probability == 1)),
                   max_band_confidence=float(confidence.max()))
    programmes = []
    for p in cfg["programme_counts"]:
        original = raw.loc[raw.programme == p]
        if original.empty:
            continue
        subset = release.loc[release.programme == p]
        original_p = original.result_band.value_counts(normalize=True).reindex(cfg["band_order"], fill_value=0)
        released_p = subset.result_band.value_counts(normalize=True).reindex(cfg["band_order"], fill_value=0) if len(subset) else pd.Series(np.nan, index=cfg["band_order"])
        absolute_error = abs(float(released_p["Below50"])-float(original_p["Below50"]))*100
        tv = float((released_p-original_p).abs().sum()/2) if len(subset) else np.nan
        programmes.append(dict(programme=p, n_original=len(original), n_released=len(subset),
                               suppression=1-len(subset)/len(original), original_below50_rate=float(original_p["Below50"]),
                               released_below50_rate=float(released_p["Below50"]), absolute_error_pp=absolute_error,
                               grade_total_variation=tv))
    programme_table = pd.DataFrame(programmes)
    covered = bool((programme_table.n_released > 0).all())
    metrics["all_programmes_retained"] = covered
    metrics["programme_below50_mae_pp"] = float(programme_table.absolute_error_pp.mean()) if covered else np.nan
    metrics["programme_below50_max_error_pp"] = float(programme_table.absolute_error_pp.max()) if covered else np.nan
    metrics["programme_grade_tv_mean"] = float(programme_table.grade_total_variation.mean()) if covered else np.nan
    metrics["max_programme_suppression"] = float(programme_table.suppression.max())
    # The following are private audit products, never inputs to attack().
    groups = sizes.rename("size").to_frame().join(distinct.rename("distinct_bands")).join(confidence.rename("max_band_confidence")).join(fail_probability.rename("below50_probability")).reset_index()
    return metrics, programme_table, groups


def worked_example():
    rows = []
    for i in range(20):
        younger = i < 10
        age = 18+(i//2) if younger else 23+(i-10)//2
        band = "Below50" if younger else ("Pass" if i % 2 == 0 else "Credit")
        rows.append(dict(student_id=f"W{i+1:02d}", age=age, postcode="9100" if i%2 == 0 else "9101",
                         programme="P01", mark={"Below50":40,"Pass":60,"Credit":70}[band], result_band=band))
    return pd.DataFrame(rows)


def validate_release(raw, release, keep, k, ell, cfg):
    assert list(release.columns) == PUBLIC
    pd.testing.assert_series_equal(release.result_band, raw.loc[keep, "result_band"], check_names=False)
    pd.testing.assert_series_equal(release.programme, raw.loc[keep, "programme"], check_names=False)
    counts = release.groupby(QI).result_band.agg(["size", "nunique"])
    assert len(release) > 0 and counts["size"].min() >= k and counts["nunique"].min() >= ell
    assert (~keep).sum() <= cfg["max_suppression"]*len(raw)+1e-9


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def plot_results(results):
    plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":10, "axes.spines.top":False,
                         "axes.spines.right":False, "figure.dpi":140})
    colours = {"k-only":"#333333", "k+l2":"#346b91", "k+l3":"#aa5936"}
    def series(ax, metric):
        for method, colour in colours.items():
            data = results.loc[(results.method == method) & results.feasible]
            stats = data.groupby("k")[metric].agg(["mean","min","max"]).reindex([5,10,20])
            mean = stats["mean"].to_numpy(float)
            ax.plot([5,10,20], mean, "o-", label=method, color=colour)
            ax.fill_between([5,10,20], stats["min"].to_numpy(float), stats["max"].to_numpy(float), color=colour, alpha=.13)
        ax.set_xticks([5,10,20]); ax.set_xlabel("Required k"); ax.grid(alpha=.18)
    fig, axes = plt.subplots(1,3,figsize=(12,3.6), layout="constrained")
    fields = [("unique_link_pct","Unique linkage (%)"), ("homogeneous_band_pct","Targets in single-band groups (%)"),
              ("modal_band_accuracy_pct","Modal result-band accuracy (%)")]
    for ax, (metric, title) in zip(axes, fields):
        series(ax, metric); ax.set_title(title)
        baseline = results.loc[results.method == "baseline", metric].mean()
        ax.axhline(baseline, color="#777777", linestyle="--", label=f"Baseline mean: {baseline:.1f}%")
        ax.set_ylim(-2,102); ax.legend(fontsize=7)
    fig.suptitle("Privacy attack results | mean and full range over five seeds", fontsize=12)
    fig.savefig(ROOT / "figures/01_privacy.png", dpi=220); plt.close(fig)
    fig, axes = plt.subplots(1,3,figsize=(12,3.6), layout="constrained")
    for ax, metric, title in zip(axes,["programme_below50_mae_pp","suppression","generalisation_loss"],
                                 ["Below-50 rate MAE (percentage points)","Overall suppression (fraction)","Age/postcode generalisation loss"]):
        series(ax,metric); ax.set_title(title); ax.set_ylim(bottom=0)
    axes[1].axhline(.10,color="#777777",linestyle="--",label="10% limit")
    axes[0].legend(fontsize=8)
    fig.suptitle("Utility and information loss | mean and full range over five seeds",fontsize=12)
    fig.savefig(ROOT / "figures/02_utility.png",dpi=220); plt.close(fig)
    selected = results.loc[(results.k == 10) & results.feasible]
    fig, axes = plt.subplots(1,2,figsize=(10,3.8),layout="constrained")
    for metric, ax, title in [("high_confidence_band_pct",axes[0],"Targets with band confidence >=80%"),
                              ("certain_binary_status_pct",axes[1],"Targets with certain below-50 / >=50 status")]:
        means = selected.groupby("method")[metric].mean().reindex(["k-only","k+l2","k+l3"])
        ax.bar(means.index,means.values,color=[colours[m] for m in means.index])
        ax.set_ylim(0,100); ax.set_ylabel("Retained targets (%)"); ax.set_title(title,fontsize=10)
        for i,v in enumerate(means.values): ax.text(i,v+1,f"{v:.1f}%",ha="center",fontsize=9)
    fig.suptitle("Residual inference risk at k=10 | averages across five seeds",fontsize=12)
    fig.savefig(ROOT / "figures/03_residual_risk.png",dpi=220); plt.close(fig)


def run():
    cfg = read_config()
    for name in ["private", "releases", "audit", "results", "figures", "worked_example"]:
        (ROOT/name).mkdir(exist_ok=True)
    pre_run = {"config_sha256":sha256(ROOT/"config.json"), "script_sha256":sha256(Path(__file__)),
               "design_version":cfg["design_version"], "generation_parameters_changed_after_results":False,
               "note":"Configuration fixed before this experiment run. Local audit record, not an externally timestamped preregistration."}
    (ROOT/"design_lock.json").write_text(json.dumps(pre_run,indent=2)+"\n")
    rows, programme_rows = [], []
    for seed in cfg["seeds"]:
        raw = generate(cfg, seed)
        raw.to_csv(ROOT/f"private/students_seed{seed}.csv",index=False)
        auxiliary = raw[["student_id"]+QI].rename(columns={"student_id":"target_label"})
        auxiliary.to_csv(ROOT/f"private/independent_target_queries_seed{seed}.csv",index=False)
        settings = [("baseline",1,1)] + [(method,k,ell) for k in cfg["k_values"] for method,ell in [("k-only",1),("k+l2",2),("k+l3",3)]]
        for setting_index,(method,k,ell) in enumerate(settings):
            stem = f"seed{seed}_{method}_k{k}"
            if method == "baseline":
                release, loss = transform(raw,cfg,0,0)
                keep = pd.Series(True,index=raw.index)
                chosen = dict(age_level=0,postcode_level=0,generalisation_loss=loss,suppression=0.0,retained=len(raw),feasible=True)
            else:
                release,keep,chosen,trace = select_release(raw,cfg,k,ell)
                trace.to_csv(ROOT/f"audit/{stem}_search.csv",index=False)
            if release is None:
                rows.append(dict(seed=seed,method=method,k=k,ell=ell,feasible=False))
                continue
            validate_release(raw,release,keep,k,ell,cfg)
            metrics,programmes,groups = evaluate(raw,release,keep,cfg)
            # Independently execute the attack for every retained target from public values.
            per_target=[]
            for idx,target in raw.loc[keep].iterrows():
                result=attack(target[QI].to_dict(),release,cfg)
                assert result["candidates"] >= k
                result.update(target_label=target.student_id,actual_band=target.result_band,
                              modal_prediction_correct=result["modal_band"] == target.result_band)
                per_target.append(result)
            attack_table=pd.DataFrame(per_target)
            assert np.isclose(100*attack_table.modal_prediction_correct.mean(),metrics["modal_band_accuracy_pct"])
            assert np.isclose(100*attack_table.unique.mean(),metrics["unique_link_pct"])
            assert np.isclose(100*(attack_table.band_confidence == 1).mean(),metrics["homogeneous_band_pct"])
            attack_table.to_csv(ROOT/f"audit/{stem}_attacks.csv",index=False)
            raw[["student_id"]].assign(retained=keep).to_csv(ROOT/f"audit/{stem}_membership.csv",index=False)
            groups.to_csv(ROOT/f"audit/{stem}_groups.csv",index=False)
            # No identifiers, index, mark, seed, row mapping or latent study context in release.
            release.sample(frac=1,random_state=seed+10000+setting_index).to_csv(ROOT/f"releases/{stem}.csv",index=False)
            rows.append({"seed":seed,"method":method,"k":k,"ell":ell,**chosen,**metrics})
            programme_rows.append(programmes.assign(seed=seed,method=method,k=k,ell=ell))
        print(f"Completed seed {seed}",flush=True)
    results=pd.DataFrame(rows)
    results.to_csv(ROOT/"results/run_metrics.csv",index=False)
    pd.concat(programme_rows,ignore_index=True).to_csv(ROOT/"results/programme_metrics.csv",index=False)
    numeric=[c for c in results.select_dtypes(include="number").columns if c not in ["seed","k","ell"]]
    results.loc[results.feasible].groupby(["method","k","ell"])[numeric].agg(["mean","min","max"]).to_csv(ROOT/"results/summary_mean_min_max.csv")
    worked=worked_example()
    worked.to_csv(ROOT/"worked_example/private_original.csv",index=False)
    illustrative=[]
    for name,k,ell in [("baseline",1,1),("k5",5,1),("k5_l2",5,2),("k5_l3",5,3)]:
        release,keep,chosen,trace=select_release(worked,cfg,k,ell)
        validate_release(worked,release,keep,k,ell,cfg)
        release.sample(frac=1,random_state=100+k+ell).to_csv(ROOT/f"worked_example/{name}_release.csv",index=False)
        metrics,_,groups=evaluate(worked,release,keep,cfg)
        groups.to_csv(ROOT/f"worked_example/{name}_groups.csv",index=False)
        illustrative.append({"method":name,"k":k,"ell":ell,**chosen,**metrics})
    pd.DataFrame(illustrative).to_csv(ROOT/"worked_example/metrics.csv",index=False)
    plot_results(results)
    environment={"python":sys.version,"platform":platform.platform(),"numpy":np.__version__,"pandas":pd.__version__,"matplotlib":matplotlib.__version__}
    (ROOT/"environment.json").write_text(json.dumps(environment,indent=2)+"\n")
    print(results.groupby(["method","k"])[["unique_link_pct","homogeneous_band_pct","modal_band_accuracy_pct","programme_below50_mae_pp","suppression"]].mean().round(4).to_string())


if __name__ == "__main__":
    run()
