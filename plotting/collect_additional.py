"""Collect missing empirical data for Figures 18/20/22/24, without editing src.

This explicit opt-in collector imports the current manual simulator read-only.
Ordinary plotting never launches simulations. Cached jobs are source-fingerprinted.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor,as_completed
from dataclasses import replace
import hashlib
import json
import math
import os
from pathlib import Path
import sys

for _name in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS"):
    os.environ[_name] = "1"


def source_fingerprint(project):
    return {str(p.relative_to(project)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((project/"src").rglob("*.py"))}


def _native(project):
    source = str(Path(project)/"src")
    if source not in sys.path:
        sys.path.insert(0,source)
    import simulation_state
    import cutoff_search
    from config import SimConfig,UserConfig,RecommenderConfig
    return simulation_state,cutoff_search,SimConfig,UserConfig,RecommenderConfig


def _stats(state):
    """Complete exploration normally; aggregate only the fixed exploitation tail."""
    _,search,_,_,_ = _native(state._plot_project)
    while not state.done and not search._exploiting(state):
        state.step()
    counts = Counter(i.phase.name.lower() for i in state.interactions)
    consumed = sum(1.-float(i.utility) for i in state.interactions)
    slate = sum(state.slate_losses)
    if not state.done:
        remaining = state.sim_config.num_steps-state.t+1
        utility = float(state.user.user_state.preference@state.recommender.estimate())
        loss = 1.-utility
        slate_loss = 1.-sum([utility]*state.recommender_config.slate_size)/state.recommender_config.slate_size
        consumed = sum([1.-float(i.utility) for i in state.interactions]+[loss]*remaining)
        slate = sum(state.slate_losses+[slate_loss]*remaining)
        counts["exploit"] += remaining
    assert sum(counts.values()) == state.sim_config.num_steps
    return {"consumed_regret":consumed,"slate_average_regret":slate,"phase_counts":dict(counts)}


def budget_episode(job):
    """Use the simulator's safe fork rule; return both regret metrics per cutoff."""
    engine,search,Sim,User,Rec = _native(job["project"])
    cfg = job["config"]
    sim = Sim(ambient_dimension=cfg["D"],num_steps=cfg["T"],seed=job["seed"],
        user_config=User(sigma_u=cfg["sigma_u"],risk=cfg["gamma_cara"]),delta_conf=cfg["delta_conf"])
    angle = math.radians(job["startup_angle"]) if job.get("startup_angle") is not None else None
    grid = sorted(set(job["budgets"]))
    rec = Rec(slate_size=cfg["K"],cutoff=max(grid),gamma_ctr=job["gamma"],startup_half_angle=angle)
    state = engine.SimulationState(sim,rec,"kracer",cutoff_after_startup=True)
    state._plot_project = job["project"]
    pending = set(grid[:-1])
    results = {}
    while not state.done and not search._exploiting(state):
        for cutoff in sorted(pending):
            if not search._can_share_next_step(state,cutoff):
                results[cutoff] = _stats(state.fork(cutoff))
                pending.remove(cutoff)
        state.step()
    results[grid[-1]] = _stats(state)
    for cutoff in sorted(pending):
        results[cutoff] = _stats(state.fork(cutoff))
    return [{"T0":budget,**results[budget]} for budget in grid]


def historical_episode(job):
    """Fresh supplied-cone runs using the current manual KRACER class directly.

    Perfect choice uses true utility and zero comparison margin; the learning
    user retains the current Bayesian-CARA model. No sphere Startup is charged.
    """
    _,_,Sim,User,Rec = _native(job["project"])
    import numpy as np
    from users.user_state import UserState
    from users.bayesian_user import BayesianUser
    from recommenders.KRACER import KRACER
    from recommenders.KRACER_state import KRACERState
    from models.interaction import Interaction
    perfect = job["model"] == "perfect"
    sim = Sim(ambient_dimension=3,num_steps=job["horizon"],seed=job["seed"],
        user_config=User(sigma_u=job["noise"]/math.sqrt(3),risk=job["risk"]),delta_conf=job["delta_conf"])
    if perfect:
        sim.h_t = lambda t:0.
    rec = Rec(slate_size=job["K"],cutoff=64,gamma_ctr=.9)
    user = BayesianUser(UserState(sim.user_config,sim))
    # Independent truth RNG leaves the user's innovation RNG identical across K/risk.
    rng = np.random.default_rng(job["seed"]+271828)
    y = rng.normal(size=2)
    y *= math.sqrt(rng.random())/np.linalg.norm(y)
    user.user_state.preference = np.r_[1.,y]/math.sqrt(1.+y@y)
    state = KRACERState(rec,sim)
    state.c = np.array([1.,0.,0.]); state.F = np.eye(3)[:,1:]; state.W = np.ones(2)
    agent = KRACER(state,sim,rec)
    consumed,slate_loss,counts = [],[],Counter()
    precision_stop = None
    while agent.t <= min(64,job["horizon"]):
        try:
            slate = agent.recommend(job["K"])
            chosen = max(slate,key=lambda item:float(user.user_state.preference@item.emb)) if perfect else user.choose(slate)
            winner = next(i for i,item in enumerate(slate) if item is chosen)
            interaction = Interaction(slate,winner,chosen,0.,0.,agent.phase,agent.t)
            user.update(interaction)
            consumed.append(1.-float(interaction.utility))
            slate_loss.append(1.-sum(float(user.user_state.preference@item.emb) for item in slate)/len(slate))
            counts[agent.phase.name.lower()] += 1
            agent.update(interaction)
        except (ValueError,FloatingPointError,np.linalg.LinAlgError) as error:
            if not perfect or math.acos(float(np.clip(user.user_state.preference@agent.estimate(),-1,1))) > 1e-7:
                raise
            precision_stop = str(error)
            break
    remaining = job["horizon"]-sum(counts.values())
    loss = 1.-float(user.user_state.preference@agent.estimate())
    counts["exploit"] += remaining
    return {"consumed_regret":sum(consumed+[loss]*remaining),
        "slate_average_regret":sum(slate_loss+[loss]*remaining),"phase_counts":dict(counts),
        "precision_stop":precision_stop,"preference":user.user_state.preference.tolist()}


def execute(job):
    return historical_episode(job) if job["kind"] == "historical" else budget_episode(job)


def _resolve(project,stored):
    marker = "manual_simulations/"
    text = str(stored).replace("\\","/")
    return project/text.split(marker,1)[1] if marker in text else Path(stored)


def plan(report,project):
    import csv
    report,project = Path(report).resolve(),Path(project).resolve()
    with (report/"data/configurations.csv").open(encoding="utf-8") as stream:
        configs = {r["case"]:r for r in csv.DictReader(stream) if r["method"] == "kracer"}
    base = configs["baseline"]
    jobs = []
    def config(row):
        return {k:(int(row[k]) if k in ("D","K","T") else float(row[k]))
                for k in ("D","K","T","sigma_u","gamma_cara","delta_conf")}
    validations = {cid:json.loads((_resolve(project,r["source_directory"])/"validation.json").read_text())
                   for cid,r in configs.items()}
    for cid,row in configs.items():
        if row["D"] != base["D"]:
            continue
        validation = validations[cid]
        for seed in validation["test_seeds"]:
            jobs.append({"kind":"budget","project":str(project),"case":cid,"seed":seed,
                "config":config(row),"gamma":float(row["value"]),"budgets":sorted({s["cutoff"] for s in validation["scores"]}),
                "selected_T0":int(row["T0"]),"split":"test"})
    v = validations["baseline"]
    grid = sorted({s["cutoff"] for s in v["scores"]})
    gammas = sorted({s["parameter"] for s in v["scores"]})
    lower = math.degrees(math.acos(1/math.sqrt(int(base["D"]))))
    for fraction in (.25,.75):
        for gamma in gammas:
            for seed in v["validation_seeds"]:
                jobs.append({"kind":"startup_validation","project":str(project),"case":f"radius_{fraction}",
                    "fraction":fraction,"startup_angle":lower+fraction*(90-lower),"seed":seed,
                    "config":config(base),"gamma":gamma,"budgets":grid,"split":"validation"})
    history = [(f"cone_K{k}_{model}",k,model,1.) for k in (2,3,5) for model in ("perfect","bayesian")]
    history += [(f"cone_risk_{risk:g}",2,"bayesian",risk) for risk in (-5.,0.,5.)]
    for cid,k,model,risk in history:
        for seed in range(26000,26000+len(v["test_seeds"])):
            jobs.append({"kind":"historical","project":str(project),"case":cid,"seed":seed,
                "model":model,"K":k,"risk":risk,"noise":float(base["normalized_noise"]),
                "horizon":int(base["T"]),"delta_conf":float(base["delta_conf"]),"T0":64,"gamma":.9,"split":"historical"})
    return {"schema":"manual_plotting_additional_v1","project":str(project),"report":str(report),
        "source_sha256":source_fingerprint(project),"configurations_sha256":hashlib.sha256((report/"data/configurations.csv").read_bytes()).hexdigest(),
        "baseline_config":config(base),"startup_test_seeds":v["test_seeds"],"jobs":jobs}


def collect(payload,cache,workers=4):
    cache = Path(cache); cache.mkdir(parents=True,exist_ok=True)
    fingerprint = hashlib.sha256(json.dumps({"src":payload["source_sha256"],"config":payload["configurations_sha256"],
        "collector":hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},sort_keys=True).encode()).hexdigest()
    completed = []
    def batch(jobs):
        pending = {}
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for job in jobs:
                key = hashlib.sha256(json.dumps([fingerprint,job],sort_keys=True).encode()).hexdigest()
                path = cache/(key+".json")
                if path.exists():
                    completed.append(json.loads(path.read_text()))
                else:
                    pending[pool.submit(execute,job)] = (path,job)
            for i,future in enumerate(as_completed(pending),1):
                path,job = pending[future]
                result = {"job":job,"result":future.result()}
                part = path.with_suffix(".part"); part.write_text(json.dumps(result,allow_nan=False)); part.replace(path)
                completed.append(result)
                if i%10 == 0 or i == len(pending):
                    print(f"Additional jobs: {i}/{len(pending)} completed",flush=True)
    batch(payload["jobs"])
    startup = [r for r in completed if r["job"]["kind"] == "startup_validation"]
    test_jobs = []
    selections = {}
    for fraction in (.25,.75):
        relevant = [r for r in startup if r["job"]["fraction"] == fraction]
        scores = {}
        for item in relevant:
            for row in item["result"]:
                scores.setdefault((row["T0"],item["job"]["gamma"]),[]).append(row["slate_average_regret"])
        cutoff,gamma = min(scores,key=lambda pair:(sum(scores[pair])/len(scores[pair]),pair[0],pair[1]))
        selections[str(fraction)] = {"T0":cutoff,"gamma":gamma}
        template = dict(relevant[0]["job"])
        for seed in payload["startup_test_seeds"]:
            test_jobs.append({**template,"kind":"startup_test","seed":seed,"gamma":gamma,"budgets":[cutoff],"split":"test"})
    batch(test_jobs)
    assert source_fingerprint(Path(payload["project"])) == payload["source_sha256"]
    return {k:v for k,v in payload.items() if k != "jobs"} | {"records":completed,"startup_selections":selections,
        "collector_fingerprint":fingerprint,"historical_protocol":{"D":3,"cone_half_angle_degrees":45,"T0":64,
        "gamma":.9,"truth":"y uniform in unit disk, u=(1,y)/sqrt(1+norm(y)^2)","fresh_manual_runs":True}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project",type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument("--report",type=Path,default=Path(__file__).resolve().parent/"results")
    parser.add_argument("--workers",type=int,default=4)
    parser.add_argument("--plan-only",action="store_true")
    args = parser.parse_args()
    payload = plan(args.report,args.project)
    destination = args.report/"additional"
    destination.mkdir(parents=True,exist_ok=True)
    (destination/"run_plan.json").write_text(json.dumps(payload,indent=2))
    print("Planned initial jobs:",len(payload["jobs"]),"plus 40 Startup test runs at validation-selected settings",flush=True)
    if not args.plan_only:
        result = collect(payload,destination/"cache",args.workers)
        (destination/"supplement.json").write_text(json.dumps(result,indent=2,allow_nan=False))
        print("Saved",destination/"supplement.json",flush=True)


if __name__ == "__main__":
    main()
