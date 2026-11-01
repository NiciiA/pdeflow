import { describe, it, expect } from "vitest";
import {
  emptyMathematics,
  importMathematics,
  insertPreset,
} from "./mathematics";
import type { Preset } from "./mathematics";
const heat: Preset = {
  id: "heat",
  label: "Heat",
  static: false,
  equations: ["u_τ=D lap(u)"],
  declarations: [
    {
      name: "u",
      kind: "field",
      rank: 0,
      status: "unknown",
      value: null,
      units: "nondimensional",
      meaning: "state",
    },
    {
      name: "D",
      kind: "parameter",
      rank: 0,
      status: "known",
      value: 0.07,
      units: "nondimensional",
      meaning: "diffusivity",
    },
  ],
};
describe("versioned mathematical state", () => {
  it("inserts into an empty draft and preserves common parameter values, measurements and boundaries", () => {
    const a = insertPreset(emptyMathematics(), heat);
    a.declarations[1].value = 0.09;
    a.measurements.rows = [{ field: "u", x: 0, y: 0, time: 0, value: 0 }];
    a.boundary.details = "my boundary note";
    const b = insertPreset(a, heat);
    expect(b.declarations[1].value).toBe(0.09);
    expect(b.measurements).toEqual(a.measurements);
    expect(b.boundary).toEqual(a.boundary);
  });
  it("roundtrips the entire request including unsupported mathematics and all information roles", () => {
    const a = insertPreset(emptyMathematics(), heat);
    a.equations = ["u_τ=D lap(u)+q"];
    a.initial.status = "partially_known";
    a.initial.details = "known observations only";
    a.observation_operator.kind = "integral";
    expect(importMathematics(JSON.stringify(a))).toEqual(a);
    expect(() => importMathematics("{bad")).toThrow();
    expect(() => importMathematics('{"schema_version":2}')).toThrow();
  });
  it("uses no fictitious physical times or initial state for static presets", () => {
    const d = insertPreset(insertPreset(emptyMathematics(), heat), {
      ...heat,
      id: "darcy",
      static: true,
      equations: ["-div(a grad(u))=f"],
    });
    expect(d.domain.times).toEqual([]);
    expect(d.initial.status).toBe("not_applicable");
  });
});
it("rejects malformed structural imports without replacing the editable request", () => {
  const a = insertPreset(emptyMathematics(), heat);
  expect(() =>
    importMathematics(JSON.stringify({ ...a, declarations: [null] })),
  ).toThrow();
  expect(() =>
    importMathematics(JSON.stringify({ ...a, preset: { bad: true } })),
  ).toThrow();
  expect(() => importMathematics("null")).toThrow();
});
