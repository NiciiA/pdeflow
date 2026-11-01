// Playground fields are uniform periodic nodes [x][y], endpoint excluded.
export type ProfileMode = "signed" | "magnitude";
export type Marginals = { x: number[]; y: number[] };
export type ProfileScales = { x: [number, number]; y: [number, number] };
export function spatialMarginals(
  field: number[][],
  lengths: number[],
  mode: ProfileMode = "signed",
): Marginals {
  const nx = field.length,
    ny = field[0]?.length;
  if (
    !nx ||
    !ny ||
    field.some((r) => r.length !== ny) ||
    lengths.length !== 2 ||
    lengths.some((l) => !Number.isFinite(l) || l <= 0)
  )
    throw new Error(
      "Profiles require a rectangular periodic field and two positive domain lengths.",
    );
  const x = Array(nx).fill(0) as number[],
    y = Array(ny).fill(0) as number[];
  field.forEach((row, i) =>
    row.forEach((value, j) => {
      if (!Number.isFinite(value))
        throw new Error("Profiles require finite field values.");
      const u = mode === "magnitude" ? Math.abs(value) : value;
      x[i] += (u * lengths[1]) / ny;
      y[j] += (u * lengths[0]) / nx;
    }),
  );
  return { x, y };
}
export function marginalScales(
  profiles: Marginals[],
  roundoffFloor = 0,
): ProfileScales {
  const result: ProfileScales = { x: [0, 0], y: [0, 0] };
  for (const profile of profiles)
    for (const axis of ["x", "y"] as const)
      for (const value of profile[axis]) {
        result[axis][0] = Math.min(result[axis][0], value);
        result[axis][1] = Math.max(result[axis][1], value);
      }
  // Keep zero visible, including identically zero fields and cancellation.
  for (const axis of ["x", "y"] as const) {
    let [lo, hi] = result[axis];
    if (hi - lo < roundoffFloor) {
      lo = -roundoffFloor / 2;
      hi = roundoffFloor / 2;
    }
    const extent = hi - lo || 1;
    result[axis] = [lo - extent * 0.04, hi + extent * 0.04];
  }
  return result;
}
// Repeat the first node at the domain end only to draw the periodic wrap.
// Integration above counts every node exactly once.
export function profilePoints(values: number[]) {
  return [...values, values[0]].map((value, i) => ({
    fraction: i / values.length,
    value,
  }));
}
