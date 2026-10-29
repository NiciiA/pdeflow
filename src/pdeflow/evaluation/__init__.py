"""Physical-space metrics and reference/observation/reconstruction/error figures."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch

from pdeflow.core import Conditioning, ProblemSpec, ReconstructionResult
from pdeflow.specialists import get_specialist
from pdeflow.observations import ObservationSet
from pdeflow.runtime import atomic_write, write_json


def metrics(result: ReconstructionResult, problem: ProblemSpec, observations: ObservationSet, reference=None):
    samples = result.samples[0].double()  # [particle,C,T,*space]
    grid = problem.grid
    if tuple(samples.shape[1:]) != (len(problem.fields), len(grid.times), *grid.shape):
        raise ValueError("Metric sample layout mismatch")
    from pdeflow.evaluation.weighted import weights,mean as weighted_mean,quantile
    population_weights=weights(result)
    mean = weighted_mean(samples,population_weights).unsqueeze(0)
    obs_predictions = observations.operator.apply(samples)
    mask = observations.mask[0].cpu()
    y = observations.values[0].cpu().double()
    errors = obs_predictions[:, mask] - y[mask]
    condition = Conditioning.from_problem(problem, len(samples))
    specialist = get_specialist(problem.family)
    fd = specialist.physics(problem, condition, "fd").evaluate(samples)
    spectral = specialist.physics(problem, condition, "spectral").evaluate(samples) if grid.temporal_kind == "transient" else None
    mass = samples.mean(tuple(range(3, samples.ndim))) * np.prod(grid.domain.lengths)
    output = {"observation_rmse": (errors.square().mean(1)*population_weights).sum().sqrt().item(),
              "measurement_noise_std": list(observations.measurement_noise_std) if observations.measurement_noise_std is not None else observations.noise_std,
              "observation_energy_scale": observations.scales(errors).tolist() if observations.measurement_noise_std is not None else max(observations.noise_std, observations.noiseless_tolerance),
              "observation_standardized_rmse": (((errors / observations.scales(errors)[mask]).square().mean(1))*population_weights).sum().sqrt().item(),
              "pde_fd_rmse": (fd.per_problem_mean_square()*population_weights).sum().sqrt().item(), "pde_spectral_space_rmse": (spectral.per_problem_mean_square()*population_weights).sum().sqrt().item() if spectral else None,
              "mass_max_drift": (mass - mass[:, :, :1]).abs().max().item() if grid.temporal_kind == "transient" else None,
              "periodic_boundary_representation": True, "boundary_residual_rmse": None,
              "periodic_seam_gradient_rms": [((samples.select(3 + j, 0) - samples.select(3 + j, grid.shape[j] - 1)) / grid.spacing[j]).square().mean().sqrt().item()
                                             for j in range(grid.dimension)],
              "boundary_note": "Periodic topology is imposed by representation. Seam gradients measure spatial variation, not boundary violations; there is no independent boundary residual",
              "ensemble_rms_spread": weighted_mean((samples-mean).square(),population_weights).mean().sqrt().item(),
              "verification": result.diagnostics.get("final_verification"), "process": result.diagnostics.get("process"),
              "cost": result.diagnostics, "provenance": result.provenance,
              "corrected": result.corrected, "stop_reason": result.stop_reason,
              "relative_reference_rms_threshold": 1e-12, "scientifically_evaluated": False}
    output['observation_by_field']={}
    for c,field in enumerate(problem.fields):
        selected=mask & (observations.operator.indices[:,0]==c)
        output['observation_by_field'][field.name]={'count':int(selected.sum()),'rmse':float((((obs_predictions[:,selected]-observations.values[0,selected]).square().mean(1))*population_weights).sum().sqrt()) if selected.any() else None}
    output.update(specialist.extra_metrics(samples, problem))
    output["population_weights"]=population_weights.tolist()
    output["population_summary"]="normalized weighted mean/spread/observation fit; PDE component validity remains worst-candidate, aggregate PDE residual is weighted; extrema/validity remain population diagnostics"
    if reference is not None:
        truth = reference.cpu().double()
        if tuple(truth.shape) != tuple(mean.shape):
            raise ValueError("Reference layout mismatch")
        unobserved = torch.ones_like(truth, dtype=torch.bool)
        ix = observations.operator.indices[mask]
        unobserved[(slice(None), *ix.unbind(1))] = False
        weights = grid.time_weights().reshape(1, 1, -1, *([1] * grid.dimension)).expand_as(truth) * grid.cell_volume

        def relative(error, selection):
            w = weights * selection
            if not torch.any(w):
                return None
            norm=(truth.square()*w).sum()
            if float(norm/w.sum())<=1e-24:return None
            return ((error.square() * w).sum() / norm).sqrt().item()

        output.update({"relative_l2_ensemble_mean": relative(mean - truth, torch.ones_like(unobserved)),
                       "relative_l2_unobserved_mean": relative(mean - truth, unobserved),
                       "relative_l2_individual": [relative(u.unsqueeze(0) - truth, unobserved) for u in samples]})
        output["fields"] = {}
        for c, field in enumerate(problem.fields):
            selection = torch.zeros_like(unobserved); selection[:,c] = True
            output["fields"][field.name] = {"relative_l2_mean": relative(mean-truth, selection), "relative_l2_unobserved": relative(mean-truth, selection & unobserved), "mae": (mean[:,c]-truth[:,c]).abs().mean().item()}
        if 'u' in output['fields']:output["field_u_mae"] = output["fields"]["u"]["mae"]
        if problem.family=='darcy':
            truth_log=truth[:,0].log();error=mean[:,0].log()-truth_log;w=weights[:,0]
            def log_relative(select):
                weight=w*select
                if not weight.any() or float((truth_log.square()*weight).sum()/weight.sum())<=1e-24:return None
                return float(((error.square()*weight).sum()/(truth_log.square()*weight).sum()).sqrt())
            output['fields']['log_a']={'relative_l2_mean':log_relative(torch.ones_like(unobserved[:,0])),'relative_l2_unobserved':log_relative(unobserved[:,0]),'mae':float(error.abs().mean())}
        lo, hi = (quantile(samples,population_weights,.025),quantile(samples,population_weights,.975)) if result.weights is not None else torch.quantile(samples, samples.new_tensor([.025, .975]), dim=0)
        output["empirical_interval_95_coverage"] = ((truth[0] >= lo) & (truth[0] <= hi)).double().mean().item()
        output["empirical_interval_95_mean_width"] = (hi - lo).mean().item()
        output["coverage_note"] = "Descriptive for this small candidate set; not posterior calibration"
        from pdeflow.evaluation.comparison import compare
        output["evaluation"]=compare(result.samples.cpu().numpy(),problem,observations,reference.cpu().numpy(),weights=population_weights.cpu().numpy())
        output["near_zero_reference_rule"]="Relative L2 undefined/null when quadrature reference RMS <=1e-12; absolute errors retained"
    return output


def _field_figure(path, result, problem, observations, reference=None):
    # A writable cache avoids user-home assumptions in notebooks and sandboxes.
    os.environ.setdefault("MPLCONFIGDIR", str(Path(path).parent / ".matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(Path(path).parent / ".cache"))
    Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grid = problem.grid
    t = len(grid.times) - 1
    from pdeflow.evaluation.weighted import weights,mean as weighted_mean
    reconstruction = weighted_mean(result.samples[0, :, 0, t],weights(result)).numpy()
    truth = None if reference is None else reference[0, 0, t].cpu().numpy()
    image_obs = np.full(grid.shape, np.nan)
    for k, ix in enumerate(observations.operator.indices):
        if ix[1] == t and observations.mask[0, k]:
            image_obs[tuple(ix[2:].tolist())] = observations.values[0, k]
    # 3D images show the actual central z plane, with axes explicitly labelled.
    suffix = ""
    if grid.dimension == 3:
        plane = grid.shape[2] // 2
        reconstruction, image_obs = reconstruction[:, :, plane], image_obs[:, :, plane]
        truth = None if truth is None else truth[:, :, plane]
        suffix = f", z={grid.coordinates()[2][plane]:.3g}"
    error = None if truth is None else np.abs(reconstruction - truth)
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.3), constrained_layout=True)
    arrays = [truth, image_obs, reconstruction, error]
    titles = ["Reference", "Measured values", "Reconstruction mean", "Absolute error"]
    available = [a for a in (truth, reconstruction) if a is not None]
    vmin, vmax = min(float(a.min()) for a in available), max(float(a.max()) for a in available)
    for j, (ax, arr, title) in enumerate(zip(axes, arrays, titles)):
        ax.set_title(title)
        ax.set_xlabel("x")
        if arr is None:
            ax.text(.5, .5, "No reference supplied", ha="center", transform=ax.transAxes)
        elif grid.dimension == 1:
            x = grid.coordinates()[0].numpy()
            ax.plot(x, arr, "." if j == 1 else "-")
            ax.set_ylabel("u" if j != 3 else "|error|")
        else:
            # Bin centres coincide with the periodic nodes (the endpoint is excluded).
            offset = 0 if grid.convention == "periodic_cell_centers" else -.5
            extent = [grid.domain.origins[0]+offset*grid.spacing[0], grid.domain.origins[0]+grid.domain.lengths[0]+offset*grid.spacing[0], grid.domain.origins[1]+offset*grid.spacing[1],grid.domain.origins[1]+grid.domain.lengths[1]+offset*grid.spacing[1]]
            im = ax.imshow(arr.T, origin="lower", extent=extent, aspect="auto", cmap="magma" if j == 3 else "coolwarm",
                           vmin=0 if j == 3 else vmin, vmax=None if j == 3 else vmax, interpolation="nearest")
            ax.set_ylabel("y")
            fig.colorbar(im, ax=ax, shrink=.75)
    fig.suptitle(f"{problem.fields[0].name}; " + (f"Physical time tau={grid.times[t]:.3g}" if grid.temporal_kind == "transient" else "Static field") + f"{suffix}; {result.provenance.get('mode', 'Fourier inverse')}")
    fig.savefig(path, dpi=130)
    plt.close(fig)


def figure(path, result, problem, observations, reference=None):
    from dataclasses import replace
    from pdeflow.observations import PointOperator
    for c, field in enumerate(problem.fields):
        select = observations.operator.indices[:,0] == c
        if not select.any():
            # Plot no observed points without inventing measurements.
            ix = observations.operator.indices[:1].clone(); ix[:,0]=0
            vals=observations.values[:,:1]; mask=torch.zeros_like(vals,dtype=torch.bool)
            # The plot helper only needs operator/values/mask.
            from types import SimpleNamespace
            obs=SimpleNamespace(operator=PointOperator(problem.grid,ix),values=vals,mask=mask)
        else:
            ix=observations.operator.indices[select].clone();ix[:,0]=0
            obs=ObservationSet(PointOperator(problem.grid,ix),observations.values[:,select],observations.mask[:,select],observations.noise_std,observations.noiseless_tolerance,measurement_noise_std=tuple(np.asarray(observations.measurement_noise_std)[select.numpy()]) if observations.measurement_noise_std is not None else None)
        target=Path(path) if c==0 else Path(path).with_name(f"reconstruction_{field.name}.png")
        _field_figure(target,replace(result,samples=result.samples[:,:,c:c+1]),replace(problem,fields=(field,)),obs,None if reference is None else reference[:,c:c+1])


def export_result(output, result, problem, observations, reference=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    report = metrics(result, problem, observations, reference)
    arrays = {"samples": result.samples.numpy()}
    previous=output/"result.npz"
    if result.inference_state is None and previous.exists():
        with np.load(previous,allow_pickle=False) as saved:
            if np.array_equal(saved['samples'],arrays['samples']):
                arrays.update({k:saved[k].copy() for k in saved.files if k.startswith('fk_') or k=='final_model_state'})
    if result.weights is not None:arrays["weights"]=result.weights.numpy()
    if hasattr(result.inference_state,"log_weights"):
        arrays.update(fk_log_weights=result.inference_state.log_weights.numpy(),fk_ancestry=result.inference_state.ancestors.numpy(),fk_previous_potential=result.inference_state.previous_potential.numpy())
    if result.inference_state is not None:
        arrays["final_model_state"]=result.inference_state.model_state.detach().cpu().numpy()
    if reference is not None:
        arrays["reference"] = reference.numpy()
    atomic_write(output / "result.npz", lambda f: np.savez_compressed(f, **arrays))
    if result.generation_snapshots is not None:
        atomic_write(output/'generation.npz',lambda f:np.savez_compressed(f,**result.generation_snapshots['arrays']))
        write_json(output/'generation.json',result.generation_snapshots['metadata'])
    write_json(output / "metrics.json", report)
    if "evaluation" in report:write_json(output/"comparison.json",report["evaluation"])
    else:(output/"comparison.json").unlink(missing_ok=True)
    write_json(output / "problem.json", problem.to_dict())
    write_json(output / "observations.json", observations.to_dict())
    figure(output / "reconstruction.png", result, problem, observations, reference)
    return report
