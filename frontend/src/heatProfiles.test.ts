import { expect, it } from "vitest";
import {
  marginalScales,
  profilePoints,
  spatialMarginals,
} from "./heatProfiles";
const field = (nx: number, ny: number, f: (x: number, y: number) => number) =>
  Array.from({ length: nx }, (_, i) =>
    Array.from({ length: ny }, (_, j) => f(i / nx, j / ny)),
  );
it("integrates signed constants with physical widths on a non-square domain/grid", () => {
  const p = spatialMarginals(
    field(7, 11, () => -3),
    [2, 5],
  );
  expect(p.x).toHaveLength(7);
  expect(p.y).toHaveLength(11);
  p.x.forEach((v) => expect(v).toBeCloseTo(-15, 12));
  p.y.forEach((v) => expect(v).toBeCloseTo(-6, 12));
});
it("integrates a separable Gaussian with finite-domain bounds, without fitting", () => {
  const n = 256,
    a = 50;
  const p = spatialMarginals(
    field(n, n, (x, y) => Math.exp(-a * ((x - 0.5) ** 2 + (y - 0.5) ** 2))),
    [2, 3],
  );
  // Integral on [-.5,.5] differs from sqrt(pi/a) by < 2e-7.
  expect(p.x[n / 2]).toBeCloseTo(3 * Math.sqrt(Math.PI / a), 6);
  expect(p.y[n / 2]).toBeCloseTo(2 * Math.sqrt(Math.PI / a), 6);
  expect(p.x[n / 4] / p.x[n / 2]).toBeCloseTo(Math.exp(-a * 0.25 ** 2), 12);
});
it("two spots retain x peaks and merge their shared y peak", () => {
  const p = spatialMarginals(
    field(
      64,
      48,
      (x, y) =>
        (Math.exp(-200 * (x - 0.25) ** 2) + Math.exp(-200 * (x - 0.75) ** 2)) *
        Math.exp(-200 * (y - 0.5) ** 2),
    ),
    [1, 1],
  );
  expect(p.x[16]).toBeGreaterThan(p.x[32] * 100);
  expect(p.x[48]).toBeCloseTo(p.x[16], 12);
  expect(p.y.indexOf(Math.max(...p.y))).toBe(24);
});
it("periodic cosine cancels across both axes, including the wrap interval", () => {
  const p = spatialMarginals(
    field(
      32,
      48,
      (x, y) => Math.cos(2 * Math.PI * x) * Math.cos(2 * Math.PI * y),
    ),
    [2, 3],
  );
  [...p.x, ...p.y].forEach((v) => expect(v).toBeCloseTo(0, 14));
  const q = spatialMarginals(
    field(32, 48, (x) => Math.cos(2 * Math.PI * x)),
    [2, 3],
  );
  expect(q.x[0]).toBeCloseTo(3, 14);
  expect(q.x[16]).toBeCloseTo(-3, 14);
  q.y.forEach((v) => expect(v).toBeCloseTo(0, 14));
});
it("magnitude is explicit, retains physical scaling and does not alter the input", () => {
  const f = [
      [-2, 2],
      [-4, 4],
    ],
    p = spatialMarginals(f, [3, 5], "magnitude");
  expect(p).toEqual({ x: [10, 20], y: [9, 9] });
  expect(spatialMarginals(f, [3, 5]).x).toEqual([0, 0]);
  expect(f[0][0]).toBe(-2);
});
it("shared interval scales retain decay and distinct reference/model amplitudes", () => {
  const reference = spatialMarginals(
    [
      [2, 2],
      [2, 2],
    ],
    [1, 1],
  );
  const model = spatialMarginals(
    [
      [-1, -1],
      [-1, -1],
    ],
    [1, 1],
  );
  const scales = marginalScales([reference, model]);
  expect(scales.x[0]).toBeLessThan(-1);
  expect(scales.x[1]).toBeGreaterThan(2);
  const decayed = spatialMarginals(
    [
      [0.5, 0.5],
      [0.5, 0.5],
    ],
    [1, 1],
  );
  expect(marginalScales([reference, model, decayed])).toEqual(scales);
});
it("draws node coordinates with endpoint repetition only for the wrap, never extra quadrature", () => {
  expect(profilePoints([1, 2, 3, 4])).toEqual([
    { fraction: 0, value: 1 },
    { fraction: 0.25, value: 2 },
    { fraction: 0.5, value: 3 },
    { fraction: 0.75, value: 4 },
    { fraction: 1, value: 1 },
  ]);
  expect(spatialMarginals([[1, 2, 3, 4]], [1, 2]).x).toEqual([5]);
});
it("does not magnify cancellation roundoff into a visible oscillating profile", () => {
  const s = marginalScales([{ x: [-1e-17, 2e-17], y: [0, 0] }], 1e-12);
  expect(s.x[1] - s.x[0]).toBeGreaterThanOrEqual(1e-12);
});
