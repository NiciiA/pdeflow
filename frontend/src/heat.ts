import type { Problem } from "./types";
export type HeatConfig = {
  family: "heat" | "wave";
  c: number;
  velocity_initial: number[][] | null;
  velocity_preset: string;
  preset: string;
  D: number;
  end: number;
  frame_count: number;
  lengths: number[];
  origins: number[];
  amplitude: number;
  background: number;
  kx: number;
  ky: number;
  phix: number;
  phiy: number;
  seed: number;
  expression: string;
  initial: number[][] | null;
  complete: string;
};
export const defaults: HeatConfig = {
  family: "heat",
  c: 0.4,
  velocity_initial: null,
  velocity_preset: "zero",
  preset: "cosine",
  D: 0.07,
  end: 1,
  frame_count: 31,
  lengths: [1, 1],
  origins: [0, 0],
  amplitude: 0.8,
  background: 0.2,
  kx: 1,
  ky: 1,
  phix: 0,
  phiy: 0,
  seed: 73,
  expression: "0.2+0.8*cos(2*pi*x)*cos(2*pi*y)",
  initial: null,
  complete: "",
};
export type Point = {
  field: string;
  x: number;
  y: number;
  time: number;
  value: number;
  sigma: number;
};
export type Reference = {
  id: string;
  problem: Problem;
  frames: number[][][];
  initial: number[][];
  initial_velocity?: number[][];
  velocity_frames?: number[][][];
  velocity_range?: number[];
  exact: number[][][] | null;
  range: number[];
  exact_max_error: number | null;
  verification: Record<string, unknown> | null;
};
export type HeatModel = {
  id: string;
  name: string;
  compatible: boolean;
  reason: string;
  pipeline_check: boolean;
  status: string;
};
export type Reconstruction = {
  frames: number[][][];
  spread: number[][][];
  velocity_frames?: number[][][];
  velocity_spread?: number[][][];
  problem: Problem;
  metrics: {
    seconds: number;
    checkpoint_sha256?: string;
    checkpoint_updates?: number;
    sensor_diagnostics?: {
      count: number;
      by_quantity: Record<
        string,
        {
          count: number;
          ensemble_mean: {
            mse: number;
            rmse: number;
            mae: number;
            noiseless_count: number;
            noisy_count: number;
          };
        }
      >;
    };
    inferred_parameters?: Record<string, number[]>;
    physics?: Record<string, unknown>;
    normalizations?: { mean: number; scale: number }[];
    field_loss_provenance?: Record<string, unknown>;
    field_normalization_scale?: number;
    observation_consistency: null | {
      rmse_against_noisy_observed_values_per_sample: number[];
    };
  };
  comparison: null | { ensemble_mean_rmse: number };
};
export async function call<T>(
  path: string,
  body?: unknown,
  signal?: AbortSignal,
): Promise<T> {
  const r = await fetch("/api" + path, {
    method: body === undefined ? "GET" : "POST",
    headers: { "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
    signal,
  });
  const v = await r.json();
  if (!r.ok) {
    const details = v.error?.details;
    throw new Error(
      v.error?.message?.startsWith("Check the highlighted") && details?.length
        ? details
            .map(
              (d: { path: string[]; message: string }) =>
                `${d.path.slice(1).join(" ")}: ${d.message}`,
            )
            .join("; ")
        : v.error?.message ||
            "The local service could not complete this action.",
    );
  }
  return v;
}
export function paint(
  field: number[][],
  x: number,
  y: number,
  radius: number,
  strength: number,
  mode: string,
) {
  const n = field.length;
  return field.map((row, i) =>
    row.map((v, j) => {
      const dx = Math.min(Math.abs(i / n - x), 1 - Math.abs(i / n - x)),
        dy = Math.min(Math.abs(j / n - y), 1 - Math.abs(j / n - y));
      const w = Math.max(0, 1 - Math.hypot(dx, dy) / radius);
      return mode === "erase"
        ? v * (1 - w)
        : v + w * strength * (mode === "remove" ? -1 : 1);
    }),
  );
}
export function bounds(frames: number[][][]) {
  let lo = Infinity,
    hi = -Infinity;
  for (const f of frames)
    for (const row of f)
      for (const v of row) {
        lo = Math.min(lo, v);
        hi = Math.max(hi, v);
      }
  return [lo, hi];
}
export function download(name: string, value: unknown) {
  const url = URL.createObjectURL(
    new Blob([JSON.stringify(value, null, 2)], { type: "application/json" }),
  );
  const a = document.createElement("a");
  a.href = url;
  a.download = name;
  a.click();
  URL.revokeObjectURL(url);
}

// Cached playback only. Scientific scrub requests use the backend Fourier/latent decoder.
export function interpolateFrames(
  frames: number[][][],
  times: number[],
  time: number,
) {
  let upper = times.findIndex((t) => t >= time);
  if (upper < 0) upper = times.length - 1;
  const lower = Math.max(0, upper - 1);
  const weight =
    upper === lower ? 0 : (time - times[lower]) / (times[upper] - times[lower]);
  return frames[lower].map((row, x) =>
    row.map((v, y) => v * (1 - weight) + frames[upper][x][y] * weight),
  );
}

export function physicalPoint(
  x: number,
  y: number,
  lengths: number[],
  origins: number[],
) {
  return [origins[0] + x * lengths[0], origins[1] + y * lengths[1]];
}

// Blank means continuous scattered times; never coerce an empty string to t=0.
export function samplingTimes(source: string) {
  return source.trim() ? source.split(",").map(Number) : [];
}
