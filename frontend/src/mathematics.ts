import type { Model, Problem, Row } from "./types";
export type InformationStatus =
  "known" | "partially_known" | "unknown" | "not_applicable";
export interface Declaration {
  name: string;
  kind: "field" | "parameter" | "coefficient" | "definition";
  rank: number;
  status: InformationStatus;
  value: number | null;
  units: string;
  meaning: string;
}
export interface MathematicalRequest {
  schema_version: 1;
  known_fields?: Record<string, unknown>;
  equations: string[];
  declarations: Declaration[];
  domain: Problem["grid"];
  boundary: { type: string; status: InformationStatus; details: string };
  initial: { status: InformationStatus; details: string };
  measurements: {
    rows: Row[];
    status: InformationStatus;
    noise_model: string;
    noise_std: number;
  };
  observation_operator: {
    kind: string;
    status: InformationStatus;
    details: string;
  };
  units: string;
  preset: string | null;
}
export interface Preset {
  id: string;
  label: string;
  equations: string[];
  declarations: Declaration[];
  static: boolean;
}
export interface ParseReport {
  routing?: {
    selected_specialist: string | null;
    selected_checkpoint: string | null;
    candidate_specialists: string[];
    compatible_checkpoints: string[];
    selection_reason: string;
    exclusions: {
      id: string;
      reason: string;
      compatibility?: { mismatches: unknown[] };
    }[];
    targets?: {
      id: string;
      state: string;
      maximum_updates: number;
      updates: number;
      command: string;
    }[];
    unknown_ranges: string[];
    range_warnings: { id: string; reason: string }[];
  };
  parsed: unknown[];
  canonical_source: string[];
  recognized_family: string;
  problem: Problem | null;
  status: Record<string, boolean>;
  reasons: string[];
}
export function emptyMathematics(
  domain?: Problem["grid"],
): MathematicalRequest {
  return {
    schema_version: 1,
    equations: [""],
    declarations: [],
    domain: structuredClone(
      domain ?? {
        shape: [8, 8],
        times: [0, 0.05, 0.1, 0.15],
        convention: "periodic_nodes_endpoint_excluded",
        domain: {
          lengths: [1, 1],
          origins: [0, 0],
          boundary: "periodic",
          geometry: "cartesian",
        },
      },
    ),
    boundary: { type: "periodic", status: "known", details: "" },
    initial: { status: "unknown", details: "" },
    measurements: {
      rows: [],
      status: "unknown",
      noise_model: "independent_gaussian",
      noise_std: 0.03,
    },
    observation_operator: {
      kind: "point_on_grid",
      status: "known",
      details: "Physical coordinates and saved physical times",
    },
    units: "nondimensional",
    preset: null,
  };
}
export function insertPreset(
  request: MathematicalRequest,
  preset: Preset,
): MathematicalRequest {
  const copy = structuredClone(request);
  copy.equations = [...preset.equations];
  copy.preset = preset.id;
  copy.declarations = preset.declarations.map((d) => {
    const existing = request.declarations.find(
      (x) => x.name === d.name && x.kind === d.kind && x.rank === d.rank,
    );
    return structuredClone(existing ?? d);
  });
  copy.domain.times = preset.static
    ? []
    : copy.domain.times.length
      ? copy.domain.times
      : [0, 0.05, 0.1, 0.15];
  copy.domain.convention =
    preset.id === "darcy"
      ? "periodic_cell_centers"
      : "periodic_nodes_endpoint_excluded";
  copy.domain.temporal_kind = preset.static ? "static" : "transient";
  delete copy.known_fields;
  copy.initial.status = preset.static
    ? "not_applicable"
    : copy.initial.status === "not_applicable"
      ? "unknown"
      : copy.initial.status;
  return copy;
}
export function checkpointLayout(
  request: MathematicalRequest,
  model: Model,
): MathematicalRequest {
  const copy = structuredClone(request);
  copy.domain = structuredClone(model.problem.grid);
  if (copy.domain.temporal_kind === "static") copy.domain.times = [];
  copy.boundary = {
    ...copy.boundary,
    type: model.problem.grid.domain.boundary,
  };
  return copy;
}
export function importMathematics(text: string): MathematicalRequest {
  const value = JSON.parse(text);
  if (!value || typeof value !== "object" || Array.isArray(value))
    throw new Error("Expected a versioned mathematical request object");
  if (
    value.schema_version !== 1 ||
    !Array.isArray(value.equations) ||
    !value.equations.every((v: unknown) => typeof v === "string") ||
    !Array.isArray(value.declarations) ||
    !Array.isArray(value.domain?.shape) ||
    !Array.isArray(value.domain?.times) ||
    !value.domain?.domain ||
    !value.boundary ||
    !value.initial ||
    !Array.isArray(value.measurements?.rows) ||
    !value.observation_operator ||
    typeof value.units !== "string" ||
    (value.preset !== null && typeof value.preset !== "string")
  )
    throw new Error(
      "Incomplete mathematical request or unsupported schema version",
    );
  const validStatuses = [
    "known",
    "partially_known",
    "unknown",
    "not_applicable",
  ];
  if (
    value.declarations.some(
      (d: Declaration) =>
        !d ||
        typeof d.name !== "string" ||
        !["field", "parameter", "coefficient", "definition"].includes(d.kind) ||
        ![0, 1].includes(d.rank) ||
        !validStatuses.includes(d.status) ||
        typeof d.units !== "string" ||
        typeof d.meaning !== "string" ||
        (d.value !== null &&
          (typeof d.value !== "number" || !Number.isFinite(d.value))),
    ) ||
    value.domain.shape.length < 1 ||
    value.domain.shape.length > 3 ||
    value.domain.shape.some(
      (n: unknown) => typeof n !== "number" || !Number.isFinite(n),
    ) ||
    !["lengths", "origins"].every(
      (k) =>
        Array.isArray(value.domain.domain[k]) &&
        value.domain.domain[k].length === value.domain.shape.length &&
        value.domain.domain[k].every(
          (n: unknown) => typeof n === "number" && Number.isFinite(n),
        ),
    ) ||
    !value.domain.times.every(
      (n: unknown) => typeof n === "number" && Number.isFinite(n),
    ) ||
    !["boundary", "initial", "observation_operator", "measurements"].every(
      (k) => validStatuses.includes(value[k].status),
    ) ||
    !["boundary", "initial", "observation_operator"].every(
      (k) => typeof value[k].details === "string",
    ) ||
    typeof value.boundary.type !== "string" ||
    typeof value.observation_operator.kind !== "string" ||
    typeof value.measurements.noise_std !== "number" ||
    value.measurements.rows.some(
      (r: unknown) => !r || typeof r !== "object" || Array.isArray(r),
    )
  )
    throw new Error(
      "Malformed request declarations, domain or information sections",
    );
  return value as MathematicalRequest;
}
