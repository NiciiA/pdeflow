"""Bounded, fixed-split heat budget ladder with validation-only selection."""
from dataclasses import replace
from pathlib import Path
import argparse
import shutil
import time

import numpy as np
import torch

from pdeflow.data import generate, load_dataset
from pdeflow.equations import heat_problem
from pdeflow.evaluation import export_result, metrics
from pdeflow.evaluation.baseline import inverse_baseline
from pdeflow.generative import make_formulation
from pdeflow.observations import sensor_observations
from pdeflow.runtime import Config, write_json
from pdeflow.sampling import sample
from pdeflow.training import load_inference, train


def cases(data, config, split, count=2):
    result = []
    for j, index in enumerate(data.indices(split)[:count]):
        p = heat_problem(config.grid, float(data.diffusivity[index]))
        truth = data.values[index:index + 1]
        seed = config.inference.seed + (1000 if split == "validation" else 2000) + j
        obs = sensor_observations(truth, config.grid, config.inference.sensors, seed, config.inference.noise_std, config.inference.noise_floor)
        result.append((data.metadata["ids"][index], p, obs, truth))
    return result


def objective(bundle, data, split):
    ix = data.indices(split)
    model = bundle.model; parameter = next(model.parameters())
    target = bundle.layout.pack(bundle.transform.encode(data.values[ix])).to(parameter)
    state, velocity, s = make_formulation(bundle.config.formulation).training_state(target, torch.Generator().manual_seed(10001), torch.Generator().manual_seed(10002))
    with torch.no_grad():
        pred = model(state, s, data.conditioning(ix).to(device=parameter.device, dtype=parameter.dtype))
    return torch.nn.functional.mse_loss(pred, velocity).item()


def budget_study(config, output, budgets=(0, 100, 500)):
    if config.data.storage != "npz":
        raise ValueError("The bounded budget-study runner supports the original small NPZ profile; use generate/train/preflight for larger sharded profiles")
    output = Path(output)
    if output.exists(): raise FileExistsError("Select a fresh study output directory")
    output.mkdir(parents=True)
    start = time.perf_counter()
    generate(config, output / "data")
    data = load_dataset(output / "data", config)
    validation, test = cases(data, config, "validation"), cases(data, config, "test")
    for split, selected in (("validation", validation), ("test", test)):
        for identifier, p, obs, truth in selected:
            write_json(output / "fixed_inputs" / split / identifier / "problem.json", p.to_dict())
            write_json(output / "fixed_inputs" / split / identifier / "observations.json", obs.to_dict())
    checkpoint = output / "training/checkpoint.pt"
    objectives = []
    for budget in budgets:
        tick = time.perf_counter()
        train(config, output / "data", output / "training", resume=checkpoint if checkpoint.exists() else None, stop_after=budget)
        destination = output / f"budget_{budget}"
        destination.mkdir()
        shutil.copyfile(output / "training/inference.pt", destination / "inference.pt")
        bundle = load_inference(destination / "inference.pt", runtime=config.runtime)
        objectives.append({"budget": budget, "training_objective": objective(bundle, data, "train"),
                           "validation_objective": objective(bundle, data, "validation"), "increment_seconds": time.perf_counter() - tick})
    # Search on validation trajectories using the largest budget, then freeze for all test budgets.
    bundle = load_inference(output / f"budget_{budgets[-1]}/inference.pt", runtime=config.runtime)
    tuning = []
    for a, b in ((.1, .001), (.3, .001), (.3, .01), (1., .01)):
        inf = replace(config.inference, observation_strength=a, physics_strength=b)
        scores = [metrics(sample(bundle, p, obs, inf, "observation_pde"), p, obs, truth)["relative_l2_unobserved_mean"] for _, p, obs, truth in validation]
        tuning.append({"observation_strength": a, "physics_strength": b, "validation_score": float(np.mean(scores))})
    selected = min(tuning, key=lambda x: x["validation_score"])
    ridge_search = []
    for ridge in (.001, .1):
        scores = [metrics(inverse_baseline(p.grid, p.known_parameters["diffusivity"], obs, config.inference.basis_degree, ridge), p, obs, truth)["relative_l2_unobserved_mean"] for _, p, obs, truth in validation]
        ridge_search.append({"ridge": ridge, "validation_score": float(np.mean(scores))})
    selected_ridge = min(ridge_search, key=lambda x: x["validation_score"])["ridge"]
    inf = replace(config.inference, observation_strength=selected["observation_strength"], physics_strength=selected["physics_strength"], ridge=selected_ridge)
    write_json(output / "selection.json", {"guidance_search": tuning, "ridge_search": ridge_search, "selected": {**selected, "ridge": selected_ridge},
               "policy": "One training trajectory; budgets are checkpoints. Validation selection at final budget, frozen across all test budgets. Fixed two cases per split, one training seed, one sampling seed."})
    reports = []
    for budget in budgets:
        bundle = load_inference(output / f"budget_{budget}/inference.pt", runtime=config.runtime)
        for identifier, p, obs, truth in test:
            # Reference-free algorithms finish before evaluation sees truth.
            results = {mode: sample(bundle, p, obs, inf, mode) for mode in ("unguided", "observation", "observation_pde")}
            results["fourier_inverse"] = inverse_baseline(p.grid, p.known_parameters["diffusivity"], obs, inf.basis_degree, inf.ridge)
            for mode, result in results.items():
                report = export_result(output / f"budget_{budget}" / identifier / mode, result, p, obs, truth)
                reports.append({"budget": budget, "case_id": identifier, "mode": mode, **report})
    write_json(output / "objectives.json", objectives)
    write_json(output / "metrics.json", reports)
    write_json(output / "study.json", {"config": config.to_dict(), "budgets": list(budgets), "seconds": time.perf_counter() - start,
               "test_ids": [c[0] for c in test], "validation_ids": [c[0] for c in validation], "scientifically_converged": False})
    figures(output, objectives, reports)
    return {"output": str(output), "budgets": list(budgets), "selected": {**selected, "ridge": selected_ridge}}


def figures(output, objectives, reports):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    for key in ("training_objective", "validation_objective"):
        ax.plot([r["budget"] for r in objectives], [r[key] for r in objectives], "o-", label=key)
    ax.set(xlabel="Optimizer updates", ylabel="Fixed-noise CFM objective"); ax.legend(); fig.savefig(output / "learning_curves.png", dpi=140); plt.close(fig)
    for xkey, xlabel, name in (("budget", "Optimizer updates", "quality_budget"), ("model_evaluations", "Batched network calls", "quality_cost")):
        fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
        for mode in ("unguided", "observation", "observation_pde", "fourier_inverse"):
            selected = [r for r in reports if r["mode"] == mode]
            grouped = []
            for budget in sorted({r["budget"] for r in selected}):
                group = [r for r in selected if r["budget"] == budget]
                x = budget if xkey == "budget" else np.mean([r["cost"]["model_evaluations"] for r in group])
                grouped.append((x, np.mean([r["relative_l2_unobserved_mean"] for r in group])))
            ax.plot(*zip(*grouped), "o-", label=mode)
        ax.set(xlabel=xlabel, ylabel="Mean unobserved relative L2 (two cases)"); ax.legend();fig.savefig(output / f"{name}.png", dpi=140);plt.close(fig)
    # Empirical spread of a real final-budget ensemble, displayed at final physical time.
    final = max(r["budget"] for r in reports)
    row = next(r for r in reports if r["budget"] == final and r["mode"] == "observation_pde")
    with np.load(output / f"budget_{final}" / row["case_id"] / "observation_pde/result.npz") as a:
        spread = a["samples"][0, :, 0, -1].std(axis=0)
    fig, ax = plt.subplots(figsize=(5, 4), constrained_layout=True)
    im=ax.imshow(spread.T, origin="lower");fig.colorbar(im, ax=ax);ax.set(title="Empirical sample spread; not calibrated uncertainty", xlabel="x node", ylabel="y node")
    fig.savefig(output / "ensemble_spread.png", dpi=140);plt.close(fig)


def main():
    p = argparse.ArgumentParser();p.add_argument("--config", required=True);p.add_argument("--output", required=True)
    args=p.parse_args();print(budget_study(Config.load(args.config), args.output))


if __name__ == "__main__":main()
