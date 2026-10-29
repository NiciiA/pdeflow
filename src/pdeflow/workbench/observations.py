"""Strict physical-coordinate import; only exact saved-time, on-grid sensors."""
import csv
import io
import math

import torch

from pdeflow.observations import ObservationSet, PointOperator


def observation_rows(observations, problem):
    rows = []
    for j, ix in enumerate(observations.operator.indices.tolist()):
        if not observations.mask[0, j]:
            continue
        row = {"field": problem.fields[ix[0]].name, "time": problem.grid.times[ix[1]], "value": float(observations.values[0, j])}
        if problem.grid.temporal_kind == "static": row.pop("time")
        for axis, name in enumerate(problem.grid.axis_names):
            row[name] = float(problem.grid.coordinates()[axis][ix[axis+2]])
        if observations.measurement_noise_std is not None:row["sigma"]=observations.measurement_noise_std[j]
        rows.append(row)
    return rows


def parse_rows(problem, rows, noise_std=.03, tolerance=.1):
    grid = problem.grid
    required = {"field", "value", *grid.axis_names} | ({"time"} if grid.temporal_kind == "transient" else set())
    if not rows or len(rows) > 20000:
        raise ValueError("Provide 1–20000 observation rows")
    indices, values, seen, sigmas = [], [], set(), []
    per_observation=any("sigma" in row for row in rows)
    fields = [f.name for f in problem.fields]
    for number, row in enumerate(rows, 1):
        if set(row)-{"sigma"} != required:
            raise ValueError(f"Row {number}: required columns are {', '.join(sorted(required))}; no extra columns")
        if row["field"] not in fields:
            raise ValueError(f"Row {number}: unknown field")
        numeric = {k: float(v) for k, v in row.items() if k != "field"}
        if not all(math.isfinite(v) for v in numeric.values()):
            raise ValueError(f"Row {number}: values must be finite")
        matches = [0] if grid.temporal_kind == "static" else [j for j, t in enumerate(grid.times) if math.isclose(t, numeric["time"], abs_tol=1e-9, rel_tol=1e-9)]
        if not matches:
            raise ValueError(f"Row {number}: time must match a saved physical time; temporal interpolation is unavailable")
        index = [fields.index(row["field"]), matches[0]]
        for axis, name in enumerate(grid.axis_names):
            x = numeric[name]
            origin, length, h = grid.domain.origins[axis], grid.domain.lengths[axis], grid.spacing[axis]
            offset = .5 if grid.convention == "periodic_cell_centers" else 0.
            node = round((x - origin) / h - offset)
            if not origin <= x < origin + length or not 0 <= node < grid.shape[axis] or not math.isclose(x, origin + (node + offset) * h, abs_tol=1e-8, rel_tol=1e-8):
                raise ValueError(f"Row {number}: {name} must be an on-grid coordinate in the endpoint-excluded domain")
            index.append(node)
        key = tuple(index)
        if key in seen:
            raise ValueError(f"Row {number}: duplicate or conflicting measurement at the same field/time/node")
        seen.add(key)
        sigma=float(row.get("sigma",noise_std))
        if not math.isfinite(sigma) or sigma<0:raise ValueError(f"Row {number}: sigma must be finite and nonnegative")
        indices.append(index); values.append(numeric["value"]);sigmas.append(sigma)
    return ObservationSet(PointOperator(grid, torch.tensor(indices, dtype=torch.long), len(fields)),
                          torch.tensor([values], dtype=torch.float64), torch.ones(1, len(rows), dtype=torch.bool), noise_std, tolerance, measurement_noise_std=tuple(sigmas) if per_observation else None)


def import_text(problem, text, format, noise_std=.03, tolerance=.1):
    if len(text.encode()) > 1_000_000:
        raise ValueError("Observation upload exceeds 1 MB")
    if format == "csv":
        rows = list(csv.DictReader(io.StringIO(text)))
    elif format == "json":
        import json
        rows = json.loads(text)
        if not isinstance(rows, list):
            raise ValueError("JSON observations must be an array of physical-coordinate rows")
    else:
        raise ValueError("Observation format must be csv or json")
    obs = parse_rows(problem, rows, noise_std, tolerance)
    return observation_rows(obs, problem)
