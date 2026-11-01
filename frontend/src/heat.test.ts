import { describe, it, expect } from "vitest";
import { paint, bounds } from "./heat";
describe("periodic starting-field brush", () => {
  it("wraps a stroke across both periodic edges, with immutable undo input", () => {
    const a = Array.from({ length: 64 }, () => Array(64).fill(0));
    const b = paint(a, 0, 0, 0.1, 1, "add");
    expect(b[0][0]).toBe(1);
    expect(b[63][0]).toBeGreaterThan(0);
    expect(b[0][63]).toBeGreaterThan(0);
    expect(b[32][32]).toBe(0);
    expect(a[0][0]).toBe(0);
  });
  it("removes or erases heat, without a live forcing state", () => {
    const a = Array.from({ length: 16 }, () => Array(16).fill(1));
    expect(paint(a, 0.5, 0.5, 0.2, 0.25, "remove")[8][8]).toBe(0.75);
    expect(paint(a, 0.5, 0.5, 0.2, 0.25, "erase")[8][8]).toBe(0);
  });
  it("retains one common physical scale rather than normalizing each image", () => {
    expect(bounds([[[0, 1]], [[2, -4]]])).toEqual([-4, 2]);
  });
});

import { physicalPoint, interpolateFrames, samplingTimes } from "./heat";
it("maps canvas fractions into physical coordinates and interpolates only cached display frames", () => {
  expect(samplingTimes(" ")).toEqual([]);
  expect(samplingTimes("0,0.237123,1")).toEqual([0, 0.237123, 1]);
  expect(physicalPoint(0.25, 0.75, [2, 0.5], [-1, 3])).toEqual([-0.5, 3.375]);
  expect(interpolateFrames([[[0]], [[1]]], [0, 1], 0.237)[0][0]).toBe(0.237);
});
