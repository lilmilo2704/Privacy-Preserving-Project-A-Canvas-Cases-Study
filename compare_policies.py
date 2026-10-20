"""Add a programme-level suppression constraint without changing original results.

Run python compare_policies.py after the original experiment. Original generator,
config, records, releases, search results and design_lock.json are read-only.
"""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from experiment import (ROOT, QI, PUBLIC, read_config, candidate, select_release,
                        validate_release, evaluate, attack, sha256)

EXT=ROOT/"policy_comparison"


def programme_suppression(raw,keep):
    """Use integer counts to enforce the cap without floating-point boundary errors."""
    counts=raw.groupby("programme").size()
    removed=raw.loc[~keep].groupby("programme").size().reindex(counts.index,fill_value=0)
    return pd.DataFrame({"n_original":counts,"n_removed":removed,"suppression":removed/counts})


def policy_candidate(raw,cfg,a,p,k,ell,programme_cap=.10):
    release,keep,info=candidate(raw,cfg,a,p,k,ell)
    by_programme=programme_suppression(raw,keep)
    cap_ok=bool((by_programme.n_removed <= programme_cap*by_programme.n_original+1e-9).all())
    info["overall_feasible"]=info["feasible"]
    info["programme_cap_feasible"]=cap_ok
    info["feasible"]=info["overall_feasible"] and cap_ok
    info["max_programme_suppression"]=float(by_programme.suppression.max())
    for programme,row in by_programme.iterrows():
        info[f"suppression_{programme}"]=float(row.suppression)
    return release,keep,info


def select_policy_release(raw,cfg,k,ell,programme_cap=.10):
    feasible=[]; traces=[]
    for a in range(len(cfg["age_widths"])):
        for p in range(len(cfg["postcode_prefix_lengths"])):
            release,keep,info=policy_candidate(raw,cfg,a,p,k,ell,programme_cap)
            traces.append(info)
            if info["feasible"]:
                key=(info["generalisation_loss"],info["suppression"],a,p)
                feasible.append((key,release,keep,info))
    if not feasible:
        return None,None,None,pd.DataFrame(traces)
    _,release,keep,chosen=min(feasible,key=lambda x:x[0])
    return release,keep,chosen,pd.DataFrame(traces)


def original_snapshot():
    # Freeze all inputs and outputs created before the extension, not only source data.
    files=[ROOT/"config.json",ROOT/"experiment.py",ROOT/"design_lock.json",ROOT/"results/run_metrics.csv",ROOT/"results/programme_metrics.csv"]
    for folder in ["private","releases","audit","worked_example"]:
        files+=sorted((ROOT/folder).glob("*.csv"))
    return {str(p.relative_to(ROOT)):sha256(p) for p in files}


def plots(combined,programme):
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"axes.spines.top":False,"axes.spines.right":False})
    colours={"overall_only":"#555555","overall_and_programme":"#346b91"}
    labels={"overall_only":"10% overall","overall_and_programme":"10% overall + per programme"}
    selected=combined[combined.method=="k+l3"]
    fig,axes=plt.subplots(2,2,figsize=(10,7),layout="constrained")
    specs=[("max_programme_suppression",100,"Worst programme suppression (%)"),
           ("programme_below50_mae_pp",1,"Programme below-50 MAE (percentage points)"),
           ("generalisation_loss",1,"Age/postcode generalisation loss"),
           ("suppression",100,"Overall suppression (%)")]
    for ax,(metric,scale,title) in zip(axes.flat,specs):
        for policy in colours:
            grouped=selected[selected.policy==policy].groupby("k")[metric].agg(["mean","min","max"]).reindex([5,10,20])*scale
            ax.plot([5,10,20],grouped["mean"],"o-",color=colours[policy],label=labels[policy])
            ax.fill_between([5,10,20],grouped["min"].to_numpy(float),grouped["max"].to_numpy(float),color=colours[policy],alpha=.13)
        ax.set_title(title,fontsize=10);ax.set_xticks([5,10,20]);ax.set_xlabel("Required k (distinct l=3)");ax.set_ylim(bottom=0);ax.grid(alpha=.15)
    axes[0,0].axhline(10,color="#999999",linestyle="--",linewidth=1)
    axes[0,0].legend(fontsize=8)
    fig.suptitle("Paired suppression-policy comparison | mean and full seed range",fontsize=13)
    fig.savefig(EXT/"figures/05_policy_comparison.png",dpi=220);plt.close(fig)

    selected=programme[(programme.method=="k+l3")&(programme.k==10)]
    fig,ax=plt.subplots(figsize=(9,4.2),layout="constrained")
    x=np.arange(6);names=list(read_config()["programme_counts"])
    for i,policy in enumerate(colours):
        grouped=selected[selected.policy==policy].groupby("programme").suppression.agg(["mean","min","max"]).reindex(names)*100
        ax.bar(x+(i-.5)*.34,grouped["mean"],.34,color=colours[policy],label=labels[policy],
               yerr=np.vstack([grouped["mean"]-grouped["min"],grouped["max"]-grouped["mean"]]),capsize=3)
    ax.set_xticks(x,[f"{p}\nN={n}" for p,n in read_config()["programme_counts"].items()]);ax.set_ylabel("Records suppressed within programme (%)")
    ax.axhline(10,color="#999999",linestyle="--",linewidth=1);ax.set_ylim(0,65)
    ax.set_title("Programme coverage at k=10, l=3 | means and full seed ranges")
    ax.legend(fontsize=9);fig.savefig(EXT/"figures/06_programme_coverage.png",dpi=220);plt.close(fig)


def run():
    cfg=read_config();policy_cfg=json.loads((EXT/"config.json").read_text())
    original_lock=json.loads((ROOT/"design_lock.json").read_text())
    assert sha256(ROOT/"config.json")==original_lock["config_sha256"]
    assert sha256(ROOT/"experiment.py")==original_lock["script_sha256"]
    assert policy_cfg["max_overall_suppression"]==cfg["max_suppression"]
    before=original_snapshot()
    for name in ["releases","audit","results","figures"]:(EXT/name).mkdir(exist_ok=True)
    extension_lock={"original_files_before":before,"extension_config_sha256":sha256(EXT/"config.json"),
                    "extension_script_sha256":sha256(Path(__file__)),"original_generator_parameters_changed":False,
                    "note":"Extension selected after inspecting original coverage findings. No new source records or generator tuning; this is an exploratory paired policy comparison."}
    (EXT/"design_lock.json").write_text(json.dumps(extension_lock,indent=2)+"\n")
    original=pd.read_csv(ROOT/"results/run_metrics.csv")
    original_programme=pd.read_csv(ROOT/"results/programme_metrics.csv")
    rows=[];programme_rows=[]
    for seed in cfg["seeds"]:
        raw=pd.read_csv(ROOT/f"private/students_seed{seed}.csv",dtype={"postcode":str})
        settings=original[(original.seed==seed)&(original.method!="baseline")]
        for index,old in settings.iterrows():
            method=old.method;k=int(old.k);ell=int(old.ell)
            stem=f"seed{seed}_{method}_k{k}"
            release,keep,chosen,trace=select_policy_release(raw,cfg,k,ell,policy_cfg["max_programme_suppression"])
            trace.to_csv(EXT/f"audit/{stem}_search.csv",index=False)
            if release is None:
                rows.append(dict(seed=seed,method=method,k=k,ell=ell,feasible=False))
                continue
            validate_release(raw,release,keep,k,ell,cfg)
            rates=programme_suppression(raw,keep)
            assert (rates.n_removed <= .10*rates.n_original+1e-9).all()
            metrics,programmes,groups=evaluate(raw,release,keep,cfg)
            assert metrics["generalisation_loss"] >= old.generalisation_loss-1e-12 if "generalisation_loss" in metrics else True
            assert chosen["generalisation_loss"] >= old.generalisation_loss-1e-12
            attacks=[]
            for _,target in raw.loc[keep].iterrows():
                result=attack(target[QI].to_dict(),release,cfg)
                assert result["candidates"]>=k
                result.update(target_label=target.student_id,actual_band=target.result_band,
                              modal_prediction_correct=result["modal_band"]==target.result_band)
                attacks.append(result)
            attacks=pd.DataFrame(attacks)
            assert np.isclose(100*attacks.modal_prediction_correct.mean(),metrics["modal_band_accuracy_pct"])
            assert np.isclose(100*attacks.unique.mean(),metrics["unique_link_pct"])
            assert np.isclose(100*(attacks.band_confidence==1).mean(),metrics["homogeneous_band_pct"])
            attacks.to_csv(EXT/f"audit/{stem}_attacks.csv",index=False)
            raw[["student_id"]].assign(retained=keep).to_csv(EXT/f"audit/{stem}_membership.csv",index=False)
            groups.to_csv(EXT/f"audit/{stem}_groups.csv",index=False)
            release.sample(frac=1,random_state=seed+20000+int(index)).to_csv(EXT/f"releases/{stem}.csv",index=False)
            rows.append({"seed":seed,"method":method,"k":k,"ell":ell,**chosen,**metrics})
            programme_rows.append(programmes.assign(seed=seed,method=method,k=k,ell=ell))
        print(f"Completed programme-cap comparison for seed {seed}",flush=True)
    new=pd.DataFrame(rows);new_programme=pd.concat(programme_rows,ignore_index=True)
    new.to_csv(EXT/"results/new_policy_metrics.csv",index=False)
    combined=pd.concat([original[original.method!="baseline"].assign(policy="overall_only"),new.assign(policy="overall_and_programme")],ignore_index=True)
    programmes=pd.concat([original_programme[original_programme.method!="baseline"].assign(policy="overall_only"),new_programme.assign(policy="overall_and_programme")],ignore_index=True)
    combined.to_csv(EXT/"results/combined_run_metrics.csv",index=False)
    programmes.to_csv(EXT/"results/combined_programme_metrics.csv",index=False)
    summary_columns=["unique_link_pct","homogeneous_band_pct","modal_band_accuracy_pct","programme_below50_mae_pp","programme_below50_max_error_pp","programme_grade_tv_mean","suppression","max_programme_suppression","generalisation_loss"]
    combined.groupby(["policy","method","k"])[summary_columns].agg(["mean","min","max"]).to_csv(EXT/"results/summary_mean_min_max.csv")
    paired=original[original.method!="baseline"].merge(new,on=["seed","method","k","ell"],suffixes=("_original","_programme_cap"),validate="one_to_one")
    for metric in summary_columns:
        paired[f"delta_{metric}"]=paired[f"{metric}_programme_cap"]-paired[f"{metric}_original"]
    paired.to_csv(EXT/"results/paired_deltas.csv",index=False)
    after=original_snapshot();assert before==after,"An original experimental artifact was modified"
    (EXT/"results/original_preservation_check.json").write_text(json.dumps({"checked_files":len(before),"all_original_hashes_unchanged":before==after,"new_runs":len(new),"paired_rows":len(paired)},indent=2)+"\n")
    plots(combined,programmes)
    print(combined[combined.method=="k+l3"].groupby(["policy","k"])[summary_columns].mean().round(5).to_string(),flush=True)


if __name__=="__main__":run()
