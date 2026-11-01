import type {
  MathematicalRequest,
  ParseReport,
  Declaration,
} from "./mathematics";
export const interpretation: Record<string, string> = {
  heat: "The field spreads over time. D controls the diffusion rate.",
  darcy:
    "Pressure is linked to spatially varying permeability and a supplied forcing field.",
  chemistry:
    "A and B react to form C, with a reverse reaction and diffusion of all three species.",
};
export function symbolName(family: string, d: Declaration) {
  const names: Record<string, Record<string, string>> = {
    heat: { u: "Solution field", D: "Diffusivity" },
    darcy: { a: "Permeability", u: "Pressure", f: "Forcing / source" },
    chemistry: {
      a: "Concentration of A",
      b: "Concentration of B",
      c: "Concentration of C",
      D_A: "Diffusivity of A",
      D_B: "Diffusivity of B",
      D_C: "Diffusivity of C",
      k_f: "Forward reaction rate",
      k_r: "Reverse reaction rate",
      r: "Net reaction rate",
    },
  };
  return names[family]?.[d.name] || d.meaning || d.name;
}
export function requestItems(
  value: MathematicalRequest,
  report: ParseReport | null,
  reviewed = false,
) {
  const known = value.declarations.filter((d) => d.status === "known");
  const fieldInputs = Object.keys(value.known_fields ?? {});
  return [
    {
      symbol: "𝓔",
      name: "Equation",
      text:
        report?.recognized_family && interpretation[report.recognized_family]
          ? report.recognized_family
          : "Editable mathematics",
      status: report?.status.parsed ? "Parsed" : "Missing / unsupported",
      section: "Problem",
      target: "equation-editor",
    },
    {
      symbol: "λ",
      name: "Parameters & forcing",
      text: `${known.length} known declarations${fieldInputs.length ? `, ${fieldInputs.join(", ")} supplied` : ""}`,
      status: known.some((d) => d.kind === "parameter" && d.value === null)
        ? "Missing value"
        : reviewed
          ? "Reviewed"
          : "Review defaults",
      section: "Problem",
      target: "physical-information",
    },
    {
      symbol: "Ω",
      name: "Domain",
      text: `${value.domain.shape.join(" × ")} · ${value.domain.shape.length}D`,
      status: reviewed ? "Reviewed" : "Review defaults",
      section: "Problem",
      target: "domain-information",
    },
    {
      symbol: "ℬ",
      name: "Boundaries",
      text:
        value.boundary.type === "periodic"
          ? "Opposite sides connect"
          : `${value.boundary.type} (execution unavailable)`,
      status:
        value.boundary.type === "periodic"
          ? reviewed
            ? "Reviewed"
            : "Review default"
          : "Incompatible",
      section: "Problem",
      target: "boundary-information",
    },
    {
      symbol: "ℐ",
      name: "Initial information",
      text:
        value.initial.status === "unknown"
          ? "Initial fields will be reconstructed"
          : value.initial.status.replaceAll("_", " "),
      status:
        value.domain.temporal_kind === "static" || !value.domain.times.length
          ? "Not applicable"
          : value.initial.status === "unknown"
            ? "Unknown target"
            : "Review information",
      section: "Problem",
      target: "initial-information",
    },
    {
      symbol: "y",
      name: "Measurements",
      text: `${value.measurements.rows.length} supplied points`,
      status: value.measurements.rows.length ? "Supplied" : "Missing",
      section: "Observations",
      target: "observation-editor",
    },
    {
      symbol: "M",
      name: "Where & how",
      text:
        value.observation_operator.kind === "point_on_grid"
          ? "Saved nodes & physical times"
          : value.observation_operator.kind,
      status:
        value.observation_operator.kind === "point_on_grid"
          ? "On-grid only"
          : "Unsupported",
      section: "Observations",
      target: "observation-editor",
    },
  ];
}
export function stable(value: unknown): string {
  if (Array.isArray(value)) return "[" + value.map(stable).join(",") + "]";
  if (value && typeof value === "object")
    return (
      "{" +
      Object.keys(value)
        .sort()
        .map(
          (k) =>
            JSON.stringify(k) +
            ":" +
            stable((value as Record<string, unknown>)[k]),
        )
        .join(",") +
      "}"
    );
  return JSON.stringify(value);
}
export function readDraft() {
  try {
    const raw = localStorage.getItem("pdeflow.draft.v2");
    if (!raw || raw.length > 1500000) return null;
    const d = JSON.parse(raw);
    return d.version === 2 && Array.isArray(d.math?.equations) ? d : null;
  } catch {
    return null;
  }
}
