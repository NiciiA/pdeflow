export type Row = Record<string, string | number>;
export interface Problem {
  schema_version: number;
  family: string;
  grid: {
    shape: number[];
    times: number[];
    temporal_kind?: string;
    convention: string;
    domain: {
      lengths: number[];
      origins: number[];
      boundary: string;
      geometry: string;
    };
  };
  fields: { name: string; units: string; role: string }[];
  known_parameters: Record<string, number>;
  known_fields?: Record<string, unknown>;
  operator_spec: unknown;
  unknown_parameters: string[];
  initial_condition: string;
}
export interface Model {
  architecture?: {
    parameters: number;
    configuration: {
      name: string;
      width: number;
      blocks: number;
      levels: number;
      conditioning_width?: number;
    };
    deepest_path_3x3_layers: number;
    resolution_policy: string;
  };
  run_id?: string;
  run_name?: string;
  export_kinds?: string[];
  steering?: { fk: boolean; sde_stepper: string };
  id: string;
  name: string;
  problem: Problem;
  config: {
    inference: Record<string, unknown>;
    provider?: Record<string, unknown> | null;
    data?: Record<string, unknown>;
    chemistry?: Record<string, unknown> | null;
    [key: string]: unknown;
  };
  training: {
    updates: number;
    diffusivity_range: number[] | null;
    status: string;
    data_mode?: string;
    evaluation_identity?: string | null;
  };
}
export interface Job {
  created_at: number;
  finished_at?: number;
  summary?: { mode: string; grid: number[]; device: string; model_id: string };
  id: string;
  status: string;
  stage: string;
  progress: number;
  seed: number;
  model_id: string;
  error?: { message: string };
  stop_reason?: string;
  process_event?: Record<string, unknown>;
  completed_steps?: number;
  total_steps?: number;
}
export interface Slice {
  native_shape?: number[];
  display_downsampled?: boolean;
  values: number[] | number[][];
  reference: number[] | number[][] | null;
  error: number[] | number[][] | null;
  shape: number[];
  axes: string[];
  coordinates: Record<string, number[]>;
  physical_time: number | null;
  slice_coordinate: number | null;
  vmin: number;
  vmax: number;
  observations: Row[];
  observation_count?: number;
  off_slice_observation_count?: number;
  units: string;
  statistic: string;
}
export interface JobResult {
  metrics: Record<string, unknown>;
  baselines?: Record<string, Record<string, unknown>>;
  request: {
    mathematical_request?: unknown;
    problem: Problem;
    observations: { indices: number[][] };
    inference: Record<string, unknown>;
    runtime: Record<string, unknown>;
    mode: string;
    model_id: string;
  };
  reference_available: boolean;
  job: Job;
}
export const backendURL =
  import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";
export class RequestError extends Error {
  details: unknown;
  constructor(message: string, details: unknown) {
    super(message);
    this.name = "RequestError";
    this.details = details;
  }
}
export async function api<T>(path: string, body?: unknown): Promise<T> {
  const response = await fetch(
    (import.meta.env.VITE_API_URL || "") + "/api" + path,
    body === undefined
      ? undefined
      : {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
        },
  ).catch(() => {
    throw new Error(
      `Backend unavailable. Expected ${backendURL}. Start it with: pdeflow serve`,
    );
  });
  const value = await response.json().catch(() => {
    throw new Error(
      `Backend unavailable. Expected ${backendURL}. Start it with: pdeflow serve`,
    );
  });
  if (!response.ok) {
    const details = value.error ?? value.detail;
    const message =
      value.error?.message ??
      (Array.isArray(value.detail)
        ? "Check the supplied fields and model selection before submitting."
        : typeof value.detail === "string"
          ? value.detail
          : "Request failed. Review the supplied inputs.");
    throw new RequestError(message, details);
  }
  return value as T;
}
export function saveFile(name: string, value: unknown) {
  const blob = new Blob(
    [typeof value === "string" ? value : JSON.stringify(value, null, 2)],
    {
      type: name.endsWith(".csv")
        ? "text/csv"
        : name.endsWith(".svg")
          ? "image/svg+xml"
          : "application/json",
    },
  );
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
export function editDimension(problem: Problem, dimension: number): Problem {
  const copy = structuredClone(problem);
  copy.grid.shape = Array.from(
    { length: dimension },
    (_, j) => copy.grid.shape[j] ?? 8,
  );
  copy.grid.domain.lengths = Array.from(
    { length: dimension },
    (_, j) => copy.grid.domain.lengths[j] ?? 1,
  );
  copy.grid.domain.origins = Array.from(
    { length: dimension },
    (_, j) => copy.grid.domain.origins[j] ?? 0,
  );
  return copy;
}
export function previewEquation(tree: unknown, depth = 0): string {
  if (depth > 16 || !tree || typeof tree !== "object")
    return "Invalid operator";
  const node = tree as {
    op: string;
    name?: string;
    value?: number;
    args?: unknown[];
  };
  const a = (node.args ?? []).map((v) => previewEquation(v, depth + 1));
  const expressions: Record<string, string> = {
    field: node.name ?? "?",
    coefficient: node.name ?? "?",
    constant: String(node.value),
    add: `(${a[0]} + ${a[1]})`,
    multiply: `${a[0]} · ${a[1]}`,
    power: `(${a[0]})^${node.value}`,
    time_derivative: `∂τ(${a[0]})`,
    laplacian: `Δ(${a[0]})`,
  };
  return (
    (expressions[node.op] ?? "Unsupported operator") +
    (depth === 0 ? " = 0" : "")
  );
}
