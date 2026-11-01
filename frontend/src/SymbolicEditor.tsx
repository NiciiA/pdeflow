import { useEffect, useRef, useState } from "react";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { TrainingDataCard } from "./TrainingDataCard";
import { MathInput } from "./MathInput";
import { interpretation, symbolName } from "./science";
import { api, saveFile } from "./types";
import type { Model, Problem } from "./types";
import {
  checkpointLayout,
  emptyMathematics,
  importMathematics,
  insertPreset,
} from "./mathematics";
import type {
  Declaration,
  InformationStatus,
  MathematicalRequest,
  ParseReport,
  Preset,
} from "./mathematics";
const statuses: InformationStatus[] = [
  "known",
  "partially_known",
  "unknown",
  "not_applicable",
];
function Status({
  value,
  onChange,
  label,
}: {
  value: InformationStatus;
  onChange: (v: InformationStatus) => void;
  label: string;
}) {
  return (
    <select
      aria-label={label}
      value={value}
      onChange={(e) => onChange(e.target.value as InformationStatus)}
    >
      {statuses.map((s) => (
        <option key={s} value={s}>
          {s.replace("_", " ")}
        </option>
      ))}
    </select>
  );
}
interface Props {
  value: MathematicalRequest;
  onChange: (v: MathematicalRequest) => void;
  models: Model[];
  modelId: string;
  onModel: (id: string) => void;
  onRoute?: (id: string) => void;
  onParsed: (p: Problem | null) => void;
  onInvalid: (reason?: string) => void;
  validated: boolean;
  onReport?: (report: ParseReport | null) => void;
}
export function SymbolicEditor({
  value,
  onChange,
  models,
  modelId,
  onModel,
  onRoute,
  onParsed,
  onInvalid,
  validated,
  onReport,
}: Props) {
  const [coverage, setCoverage] = useState<
    {
      id: string;
      specialist: string;
      grid: Problem["grid"];
      state: string;
      updates: number;
      maximum_updates: number;
      command: string;
    }[]
  >([]);
  const [chosenLayout, setTargetLayout] = useState("heat-2d-512");
  useEffect(() => {
    let live = true;
    void api<{ targets: typeof coverage }>("/training-coverage")
      .then((v) => {
        if (live) setCoverage(Array.isArray(v.targets) ? v.targets : []);
      })
      .catch(() => {});
    return () => {
      live = false;
    };
  }, []);
  const [trainingChoices, setTrainingChoices] = useState<
    {
      name: string;
      specialist: string;
      grid: number[];
      times: number[];
      temporal_kind: string;
    }[]
  >([]);
  useEffect(() => {
    let live = true;
    void api<typeof trainingChoices>("/training-presets")
      .then((v) => {
        if (live) setTrainingChoices(Array.isArray(v) ? v : []);
      })
      .catch(() => {});
    return () => {
      live = false;
    };
  }, []);
  const [catalog, setCatalog] = useState<Preset[]>([]),
    [search, setSearch] = useState(""),
    [report, setReport] = useState<ParseReport | null>(null),
    [error, setError] = useState(""),
    [advanced, setAdvanced] = useState<string | null>(() =>
      localStorage.getItem("pdeflow.raw-request.v1"),
    ),
    [source, setSource] = useState(false),
    [pending, setPending] = useState<MathematicalRequest | null>(null),
    [parsing, setParsing] = useState(false),
    [advancedInvalid, setAdvancedInvalid] = useState(() => {
      const raw = localStorage.getItem("pdeflow.raw-request.v1");
      if (!raw) return false;
      try {
        importMathematics(raw);
        return false;
      } catch {
        return true;
      }
    });
  const sequence = useRef(0),
    latest = useRef(value),
    callbacks = useRef({ onParsed, onInvalid, onRoute, onReport });
  useEffect(() => {
    callbacks.current = { onParsed, onInvalid, onRoute, onReport };
  }, [onParsed, onInvalid, onRoute, onReport]);
  useEffect(() => {
    void api<Preset[]>("/mathematics/presets")
      .then(setCatalog)
      .catch((e) => setError(String(e)));
  }, []);
  const key = JSON.stringify(value);
  useEffect(() => {
    latest.current = value;
    const version = ++sequence.current;
    let live = true;
    callbacks.current.onInvalid();
    // Preserved invalid external source must immediately invalidate executable state.
    if (advancedInvalid) {
      callbacks.current.onInvalid(
        "Your preserved request JSON is incomplete or invalid. Correct it in the equation editor before running.",
      );
      // eslint-disable-next-line react-hooks/set-state-in-effect
      setReport(null);
      callbacks.current.onReport?.(null);
      setError(
        "The preserved request JSON draft is incomplete or invalid. Correct it or edit/reset the draft before execution.",
      );
      return;
    }
    const timer = setTimeout(() => {
      // A manual validation may have superseded this pending debounce.
      if (!live || version !== sequence.current) return;
      setParsing(true);
      setReport(null);
      callbacks.current.onReport?.(null);
      void api<ParseReport>("/mathematics/parse", {
        request: value,
        model_id: modelId || null,
      })
        .then((r) => {
          if (!live || version !== sequence.current) return;
          setReport(r);
          callbacks.current.onReport?.(r);
          setError("");
          callbacks.current.onParsed(r.problem);
          callbacks.current.onRoute?.(r.routing?.selected_checkpoint ?? "");
        })
        .catch((e) => {
          if (!live || version !== sequence.current) return;
          setError(String(e));
          setReport(null);
          callbacks.current.onInvalid(
            `Current equation validation failed: ${String(e)}`,
          );
          callbacks.current.onParsed(null);
        })
        .finally(() => {
          if (live && version === sequence.current) setParsing(false);
        });
    }, 250);
    return () => {
      clearTimeout(timer);
      live = false;
    };
    // JSON identity avoids reparsing unchanged values on parent renders.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key, modelId, advancedInvalid]);
  useEffect(() => {
    try {
      if (advanced !== null)
        localStorage.setItem("pdeflow.raw-request.v1", advanced);
      else localStorage.removeItem("pdeflow.raw-request.v1");
    } catch {
      /* Complete request export remains available if storage is full. */
    }
  }, [advanced]);
  const matchingTargets = coverage.filter(
    (t) =>
      !report?.recognized_family ||
      report.recognized_family === "custom" ||
      t.specialist === report.recognized_family,
  );
  const targetLayout = matchingTargets.some((t) => t.id === chosenLayout)
    ? chosenLayout
    : (matchingTargets[0]?.id ?? chosenLayout);
  function change(next: MathematicalRequest) {
    // MathLive can emit multiple input notifications for one keyboard edit.
    // An identical notification must not cancel the parse without replacing it.
    if (JSON.stringify(next) === JSON.stringify(latest.current)) return;
    latest.current = next;
    sequence.current++;
    onInvalid();
    setReport(null);
    setError("");
    setAdvanced(null);
    setAdvancedInvalid(false);
    onChange(next);
  }
  function declaration(index: number, patch: Partial<Declaration>) {
    change({
      ...value,
      declarations: value.declarations.map((d, i) =>
        i === index ? { ...d, ...patch } : d,
      ),
    });
  }
  function propose(next: MathematicalRequest) {
    // A blank equation can still contain user declarations or initial data.
    // Only the untouched default request counts as an empty insertion.
    if (JSON.stringify(value) !== JSON.stringify(emptyMathematics()))
      setPending(next);
    else change(next);
  }
  const model = models.find(
      (m) => m.id === (modelId || report?.routing?.selected_checkpoint),
    ),
    dimension = value.domain.shape.length;
  const matchingTraining = trainingChoices.find(
    (c) =>
      c.specialist === report?.recognized_family &&
      JSON.stringify(c.grid) === JSON.stringify(value.domain.shape) &&
      (c.temporal_kind === "static"
        ? value.domain.times.length === 0
        : JSON.stringify(c.times) === JSON.stringify(value.domain.times)) &&
      value.domain.domain.lengths.every((v) => v === 1) &&
      value.domain.domain.origins.every((v) => v === 0) &&
      value.domain.domain.boundary === "periodic",
  );
  return (
    <div className="stack symbolic-editor" id="equation-editor">
      <Card>
        <CardHeader>
          <CardTitle>Editable mathematics</CardTitle>
          <CardDescription>
            Complete mathematical reconstruction request. Physical time is τ;
            generation time is configured separately. Presets never start
            training.
          </CardDescription>
        </CardHeader>
        <CardContent className="stack">
          <h2>𝓔 — Equation or coupled system</h2>
          <details
            className="equation-presets"
            open={!value.equations.some((e) => e.trim())}
          >
            <summary>Insert an editable equation / search presets</summary>
            <Input
              aria-label="Search equation presets"
              placeholder="Optional preset search"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
            />
            <div className="preset-catalog">
              {catalog
                .filter((p) =>
                  p.label.toLowerCase().includes(search.toLowerCase()),
                )
                .map((p) => (
                  <Button
                    key={p.id}
                    variant="outline"
                    onClick={() => propose(insertPreset(value, p))}
                  >
                    {p.label}
                    <small>
                      {p.id === "heat"
                        ? "Fourier data generator + heat specialist"
                        : p.id === "darcy"
                          ? "Conservative FV data + Darcy specialist"
                          : p.id === "chemistry"
                            ? "Positive reaction–diffusion + chemistry specialist"
                            : "editable; data generator unavailable"}
                    </small>
                  </Button>
                ))}
            </div>
          </details>
          <p>
            Draft: {value.preset ?? "custom / empty"}. Only the backend
            canonical equation determines execution support.
          </p>
          {value.equations.map((equation, index) => (
            <div className="equation-row" key={index}>
              <label>Equation {index + 1}</label>
              <MathInput
                index={index}
                names={value.declarations.map((d) => d.name)}
                value={equation}
                onChange={(text) =>
                  change({
                    ...latest.current,
                    equations: latest.current.equations.map((e, i) =>
                      i === index ? text : e,
                    ),
                    preset: null,
                  })
                }
              />
              {value.equations.length > 1 && (
                <Button
                  variant="ghost"
                  onClick={() =>
                    change({
                      ...value,
                      equations: value.equations.filter((_, i) => i !== index),
                      preset: null,
                    })
                  }
                >
                  Remove equation {index + 1}
                </Button>
              )}
              {source && (
                <Textarea
                  aria-label={`LaTeX source ${index + 1}`}
                  value={equation}
                  onChange={(e) =>
                    change({
                      ...value,
                      equations: value.equations.map((v, i) =>
                        i === index ? e.target.value : v,
                      ),
                      preset: null,
                    })
                  }
                />
              )}
            </div>
          ))}
          <div className="button-row">
            <Button
              variant="outline"
              onClick={() =>
                change({
                  ...value,
                  equations: [...value.equations, ""],
                  preset: null,
                })
              }
              disabled={value.equations.length >= 8}
            >
              Add equation
            </Button>
            <Button variant="ghost" onClick={() => setSource(!source)}>
              {source ? "Hide" : "Show"} LaTeX source
            </Button>
          </div>
          <p className="validation-note">
            {parsing
              ? "Parsing current draft…"
              : report
                ? "Current source parsed"
                : "Current source is not parsed"}
            . Incomplete input is preserved.
          </p>
          {error && <p role="alert">{error}</p>}
          <div className="support-status">
            {[
              "editable",
              "parsed",
              "residual",
              "data_generator",
              "checkpoint",
              "reconstruction",
            ].map((s) => (
              <span key={s}>
                {s.replace("_", " ")}:{" "}
                {s === "editable"
                  ? "yes"
                  : s === "reconstruction" && validated
                    ? "yes"
                    : report?.status[s]
                      ? "yes"
                      : s === "reconstruction"
                        ? "validate in Reconstruction"
                        : "unavailable"}
              </span>
            ))}
          </div>
          <p className="equation-interpretation">
            {report && interpretation[report.recognized_family]
              ? interpretation[report.recognized_family]
              : "The draft remains editable. A parsed expression needs a matching executable specialist before reconstruction."}
          </p>
          {report?.reasons.map((r) => (
            <p key={r} className="validation-note">
              {r}
            </p>
          ))}
        </CardContent>
      </Card>
      <Card className="symbols-panel">
        <CardHeader>
          <CardTitle>Symbols in this problem</CardTitle>
          <CardDescription>
            Physical parameters λ are separate from numerical guidance weights.
            Values below are the current request; preset defaults still need
            review.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="bounded-table">
            <table aria-label="Symbols in this problem">
              <thead>
                <tr>
                  <th>Symbol / plain name</th>
                  <th>Role</th>
                  <th>Value or field</th>
                  <th>Units</th>
                </tr>
              </thead>
              <tbody>
                {value.declarations.map((d, i) => (
                  <tr key={i}>
                    <td>
                      <strong>{d.name}</strong>
                      <small>
                        {symbolName(report?.recognized_family ?? "", d)}
                      </small>
                    </td>
                    <td>
                      {d.kind} · {d.status.replaceAll("_", " ")}
                    </td>
                    <td>
                      {d.kind === "parameter" && d.status === "known" ? (
                        <Input
                          aria-label={
                            d.name === "D" ? "Diffusivity" : `Value ${d.name}`
                          }
                          type="number"
                          value={d.value ?? ""}
                          onChange={(e) =>
                            declaration(i, {
                              value:
                                e.target.value === ""
                                  ? null
                                  : Number(e.target.value),
                            })
                          }
                        />
                      ) : Object.hasOwn(value.known_fields ?? {}, d.name) ? (
                        "Supplied spatial array"
                      ) : d.kind === "field" || d.kind === "coefficient" ? (
                        d.status === "unknown" ? (
                          "To reconstruct"
                        ) : (
                          "Spatial input / see preview"
                        )
                      ) : d.kind === "definition" ? (
                        "Equation-defined"
                      ) : (
                        (d.value ?? "Missing")
                      )}
                    </td>
                    <td>{d.units || "Not specified"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          {!value.declarations.length && (
            <p>Insert or declare symbols to see their physical roles here.</p>
          )}
          {value.declarations.some(
            (d) => d.name === "f" && d.status === "known",
          ) && (
            <div className="forcing-primary">
              <p>
                <strong>Known forcing f(x)</strong> —{" "}
                {value.known_fields?.f
                  ? "Spatial array supplied. Inspect it in the input preview."
                  : "A numeric spatial forcing field is missing."}{" "}
                It is an input, not pressure/permeability truth.
              </p>
              <Button
                variant="outline"
                onClick={() => {
                  const build = (axis: number, indices: number[]): unknown =>
                    axis === dimension
                      ? indices.reduce(
                          (v, k, j) =>
                            v +
                            Math.sin(
                              (2 * Math.PI * (k + 0.5)) / value.domain.shape[j],
                            ),
                          0,
                        )
                      : Array.from(
                          { length: value.domain.shape[axis] },
                          (_, k) => build(axis + 1, [...indices, k]),
                        );
                  change({ ...value, known_fields: { f: build(0, []) } });
                }}
              >
                Set known mean-zero sinusoidal forcing
              </Button>
              <label className="file-input-label">
                Upload known forcing array (JSON)
                <Input
                  aria-label="Upload known forcing array"
                  type="file"
                  accept=".json"
                  onChange={(e) => {
                    const file = e.target.files?.[0];
                    if (file)
                      void file.text().then((text) => {
                        try {
                          change({
                            ...value,
                            known_fields: {
                              ...value.known_fields,
                              f: JSON.parse(text),
                            },
                          });
                        } catch {
                          setError(
                            "Forcing JSON is invalid; retain and correct your file.",
                          );
                        }
                      });
                    e.target.value = "";
                  }}
                />
              </label>
            </div>
          )}
        </CardContent>
      </Card>
      <Card>
        <CardHeader>
          <CardTitle>Training target coverage and compatible layouts</CardTitle>
          <CardDescription>
            Review the exact native domain and saved times before adopting a
            layout. This never trains or silently replaces physics or
            measurements.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <select
            aria-label="Training target layout"
            value={targetLayout}
            onChange={(e) => setTargetLayout(e.target.value)}
          >
            {coverage
              .filter(
                (t) =>
                  !report?.recognized_family ||
                  report.recognized_family === "custom" ||
                  t.specialist === report.recognized_family,
              )
              .map((t) => (
                <option key={t.id} value={t.id}>
                  {t.id} · {t.grid.shape.join("×")} · {t.state} ·{" "}
                  {t.maximum_updates}/{t.updates}
                </option>
              ))}
          </select>
          {coverage.find((t) => t.id === targetLayout) && (
            <>
              <pre>
                {JSON.stringify(
                  coverage.find((t) => t.id === targetLayout)?.grid,
                  null,
                  2,
                )}
              </pre>
              <code>
                {coverage.find((t) => t.id === targetLayout)?.command}
              </code>
            </>
          )}
          <Button
            variant="outline"
            disabled={!coverage.some((t) => t.id === targetLayout)}
            onClick={() => {
              const t = coverage.find((t) => t.id === targetLayout);
              if (!t) return;
              const next = structuredClone(value);
              next.domain = structuredClone(t.grid);
              if (t.grid.temporal_kind === "static") next.domain.times = [];
              propose(next);
            }}
          >
            Use model-compatible layout
          </Button>
          <Button
            variant="outline"
            disabled={
              !coverage.some((t) => t.id === "heat-2d-512") ||
              !catalog.some((p) => p.id === "heat")
            }
            onClick={() => {
              const t = coverage.find((t) => t.id === "heat-2d-512"),
                heat = catalog.find((p) => p.id === "heat");
              if (t && heat) {
                const next = insertPreset(value, heat);
                next.domain = structuredClone(t.grid);
                propose(next);
              }
            }}
          >
            Load 512² heat example
          </Button>
          <p>
            Known parameter values remain conditioning, within the recorded
            training ranges. Existing measurements are preserved and must be
            revalidated against a changed layout. The 512² heat layout is
            periodic [0,1]², τ=[0,.05,.1]; the initial field remains unknown to
            inference.
          </p>
          {report?.routing?.targets?.map((t) => (
            <p key={t.id}>
              {t.id}: {t.state}, {t.maximum_updates}/{t.updates} updates ·{" "}
              <code>{t.command}</code>
            </p>
          ))}
        </CardContent>
      </Card>
      <Card className="automatic-model">
        <CardHeader>
          <CardTitle>Compatible model</CardTitle>
          <CardDescription>
            Selection follows the current parsed equation and full request,
            never the preset name.
          </CardDescription>
        </CardHeader>
        <CardContent>
          {model && report?.status.checkpoint ? (
            <>
              <p>
                <strong>{model.name}</strong>
              </p>
              <p>
                {model.problem.grid.shape.join(" × ")} ·{" "}
                {model.problem.grid.temporal_kind === "static"
                  ? "Static pressure / coefficient fields"
                  : `${model.problem.grid.times.length} saved physical times`}{" "}
                · {model.training.status}
              </p>
              <p>
                Fixed layout and supported boundaries only. Recorded training
                ranges may be unknown; no confidence or accuracy rating is
                implied.
              </p>
            </>
          ) : (
            <>
              <p>
                No compatible model selected.{" "}
                {report?.routing?.exclusions[0]?.reason ||
                  report?.reasons[0] ||
                  "Insert a supported equation and review its layout."}
              </p>
              {report &&
              ["heat", "darcy", "chemistry"].includes(
                report.recognized_family,
              ) &&
              [2, 3].includes(dimension) ? (
                <p>
                  {matchingTraining ? (
                    <>
                      Layout-compatible training profile:{" "}
                      <code>
                        pdeflow train --preset {matchingTraining.name} --device{" "}
                        {matchingTraining.name.includes("cuda")
                          ? "cuda"
                          : "mps"}
                      </code>
                      . Check coefficient ranges and resource requirements
                      before training.
                    </>
                  ) : (
                    <>
                      No packaged training preset matches this exact grid,
                      domain and saved-time layout. Keep the requested
                      resolution and create a matching configuration;{" "}
                      <code>pdeflow presets</code> lists actual layouts. See
                      docs/capacity-resolution.md. Reference solving remains
                      available without neural weights.
                    </>
                  )}
                </p>
              ) : (
                <p>
                  <code>pdeflow presets</code> lists implemented training
                  profiles. Unsupported equations need a new specialist; a
                  generic heat model cannot execute them.
                </p>
              )}
            </>
          )}
          <Button
            variant="outline"
            onClick={() => {
              const detail = document.getElementById(
                "physical-information",
              ) as HTMLDetailsElement;
              if (detail) detail.open = true;
              document
                .getElementById("model-override")
                ?.scrollIntoView({ behavior: "smooth" });
            }}
          >
            Change model / review layout
          </Button>
        </CardContent>
      </Card>
      <details id="physical-information" className="expert-problem">
        <summary>
          Edit declarations, domain, conditions & model override
        </summary>
        <div className="two-columns">
          <div className="stack">
            <Card>
              <CardHeader>
                <CardTitle>
                  λ — Physical parameters, coefficients and fields
                </CardTitle>
                <CardDescription>
                  Declare every symbol. Numerical guidance weights are separate
                  controls in Reconstruction. Vector components follow x/y/z
                  order.
                </CardDescription>
              </CardHeader>
              <CardContent className="stack">
                {value.declarations.map((d, i) => (
                  <div className="declaration" key={i}>
                    <div className="form-pair">
                      <label>
                        Symbol
                        <Input
                          aria-label={`Symbol ${i + 1}`}
                          value={d.name}
                          onChange={(e) =>
                            declaration(i, { name: e.target.value })
                          }
                        />
                      </label>
                      <label>
                        Meaning
                        <Input
                          aria-label={`Meaning ${d.name}`}
                          value={d.meaning}
                          onChange={(e) =>
                            declaration(i, { meaning: e.target.value })
                          }
                        />
                      </label>
                    </div>
                    <div className="form-pair">
                      <select
                        aria-label={`Kind ${d.name}`}
                        value={d.kind}
                        onChange={(e) =>
                          declaration(i, {
                            kind: e.target.value as Declaration["kind"],
                          })
                        }
                      >
                        {[
                          "field",
                          "parameter",
                          "coefficient",
                          "definition",
                        ].map((k) => (
                          <option key={k}>{k}</option>
                        ))}
                      </select>
                      <select
                        aria-label={`Rank ${d.name}`}
                        value={d.rank}
                        onChange={(e) =>
                          declaration(i, { rank: Number(e.target.value) })
                        }
                      >
                        <option value={0}>Scalar</option>
                        <option value={1}>
                          Vector ({dimension} components)
                        </option>
                      </select>
                    </div>
                    <Status
                      label={`Status ${d.name}`}
                      value={d.status}
                      onChange={(status) => declaration(i, { status })}
                    />
                    <label>
                      Value {d.name}
                      <Input
                        aria-label={
                          d.name === "D"
                            ? "Expert Diffusivity"
                            : `Expert value ${d.name}`
                        }
                        type="number"
                        value={d.value ?? ""}
                        onChange={(e) =>
                          declaration(i, {
                            value:
                              e.target.value === ""
                                ? null
                                : Number(e.target.value),
                          })
                        }
                      />
                    </label>
                    <label>
                      Units / scale
                      <Input
                        aria-label={`Units ${d.name}`}
                        value={d.units}
                        onChange={(e) =>
                          declaration(i, { units: e.target.value })
                        }
                      />
                    </label>
                    <Button
                      variant="ghost"
                      onClick={() =>
                        change({
                          ...value,
                          declarations: value.declarations.filter(
                            (_, j) => j !== i,
                          ),
                        })
                      }
                    >
                      Remove {d.name}
                    </Button>
                  </div>
                ))}
                <Button
                  variant="outline"
                  onClick={() =>
                    change({
                      ...value,
                      declarations: [
                        ...value.declarations,
                        {
                          name: "q",
                          kind: "parameter",
                          rank: 0,
                          status: "known",
                          value: 1,
                          units: "nondimensional",
                          meaning: "Declare meaning",
                        },
                      ],
                    })
                  }
                >
                  Add declaration
                </Button>
                {value.declarations.some(
                  (d) => d.name === "f" && d.status === "known",
                ) && (
                  <div className="stack">
                    <label>
                      Known forcing f (cell array JSON)
                      <Textarea
                        aria-label="Known forcing f"
                        value={JSON.stringify(value.known_fields?.f ?? null)}
                        onChange={(e) => {
                          try {
                            change({
                              ...value,
                              known_fields: { f: JSON.parse(e.target.value) },
                            });
                            setError("");
                          } catch {
                            setError(
                              "Forcing requires valid JSON; use the complete request editor for incomplete edits",
                            );
                          }
                        }}
                      />
                    </label>
                    <Button
                      variant="outline"
                      onClick={() => {
                        const build = (
                          axis: number,
                          indices: number[],
                        ): unknown =>
                          axis === dimension
                            ? indices.reduce(
                                (v, k, j) =>
                                  v +
                                  Math.sin(
                                    (2 * Math.PI * (k + 0.5)) /
                                      value.domain.shape[j],
                                  ),
                                0,
                              )
                            : Array.from(
                                { length: value.domain.shape[axis] },
                                (_, k) => build(axis + 1, [...indices, k]),
                              );
                        change({ ...value, known_fields: { f: build(0, []) } });
                      }}
                    >
                      Replace forcing with the sinusoidal example
                    </Button>
                    <p>
                      This explicitly supplies f; it does not supply hidden
                      permeability or pressure.
                    </p>
                  </div>
                )}
                <label>
                  Request units / nondimensionalization
                  <Input
                    aria-label="Request units"
                    value={value.units}
                    onChange={(e) =>
                      change({ ...value, units: e.target.value })
                    }
                  />
                </label>
                <p>
                  Unknowns:{" "}
                  {value.declarations
                    .filter(
                      (d) =>
                        d.status === "unknown" ||
                        d.status === "partially_known",
                    )
                    .map((d) => `${d.name} (${d.kind})`)
                    .join(", ") || "none declared"}
                </p>
              </CardContent>
            </Card>
            <Card id="domain-information">
              <CardHeader>
                <CardTitle>Ω — Spatial domain and discretization</CardTitle>
              </CardHeader>
              <CardContent className="stack">
                <label>
                  Spatial dimension
                  <select
                    aria-label="Spatial dimension"
                    value={dimension}
                    onChange={(e) => {
                      const n = Number(e.target.value);
                      change({
                        ...value,
                        domain: {
                          ...value.domain,
                          shape: Array.from(
                            { length: n },
                            (_, j) => value.domain.shape[j] ?? 8,
                          ),
                          domain: {
                            ...value.domain.domain,
                            lengths: Array.from(
                              { length: n },
                              (_, j) => value.domain.domain.lengths[j] ?? 1,
                            ),
                            origins: Array.from(
                              { length: n },
                              (_, j) => value.domain.domain.origins[j] ?? 0,
                            ),
                          },
                        },
                      });
                    }}
                  >
                    {[1, 2, 3].map((d) => (
                      <option key={d} value={d}>
                        {d}D
                      </option>
                    ))}
                  </select>
                </label>
                {value.domain.shape.map((size, j) => (
                  <div className="axis-row" key={j}>
                    <strong>{["x", "y", "z"][j]}</strong>
                    {(["origins", "lengths"] as const).map((k) => (
                      <label key={k}>
                        {k}
                        <Input
                          aria-label={`${["x", "y", "z"][j]} ${k}`}
                          type="number"
                          value={value.domain.domain[k][j]}
                          onChange={(e) => {
                            const next = structuredClone(value);
                            next.domain.domain[k][j] = Number(e.target.value);
                            change(next);
                          }}
                        />
                      </label>
                    ))}
                    <label>
                      Nodes
                      <Input
                        aria-label={`${["x", "y", "z"][j]} nodes`}
                        type="number"
                        value={size}
                        onChange={(e) => {
                          const next = structuredClone(value);
                          next.domain.shape[j] = Number(e.target.value);
                          change(next);
                        }}
                      />
                    </label>
                  </div>
                ))}
                <label>
                  Grid convention
                  <select
                    aria-label="Grid convention"
                    value={value.domain.convention}
                    onChange={(e) =>
                      change({
                        ...value,
                        domain: { ...value.domain, convention: e.target.value },
                      })
                    }
                  >
                    <option value="periodic_nodes_endpoint_excluded">
                      Periodic nodes
                    </option>
                    <option value="periodic_cell_centers">
                      Periodic cell centres
                    </option>
                  </select>
                </label>
                <label>
                  Physical times τ (empty for static equations)
                  <Input
                    aria-label="Physical times"
                    value={value.domain.times.join(", ")}
                    onChange={(e) =>
                      change({
                        ...value,
                        domain: {
                          ...value.domain,
                          times: e.target.value.trim()
                            ? e.target.value.split(",").map(Number)
                            : [],
                        },
                      })
                    }
                  />
                </label>
                <label>
                  Geometry
                  <select
                    aria-label="Geometry"
                    value={value.domain.domain.geometry}
                    onChange={(e) =>
                      change({
                        ...value,
                        domain: {
                          ...value.domain,
                          domain: {
                            ...value.domain.domain,
                            geometry: e.target.value,
                          },
                        },
                      })
                    }
                  >
                    <option value="cartesian">Cartesian box</option>
                    <option value="irregular">
                      Irregular (execution unavailable)
                    </option>
                  </select>
                </label>
              </CardContent>
            </Card>
          </div>
          <div className="stack">
            <Card id="boundary-information">
              <CardHeader>
                <CardTitle>ℬ — Boundary information</CardTitle>
              </CardHeader>
              <CardContent className="stack">
                <Status
                  label="Boundary information status"
                  value={value.boundary.status}
                  onChange={(status) =>
                    change({
                      ...value,
                      boundary: { ...value.boundary, status },
                    })
                  }
                />
                <select
                  aria-label="Boundary convention"
                  value={value.boundary.type}
                  onChange={(e) =>
                    change({
                      ...value,
                      boundary: { ...value.boundary, type: e.target.value },
                      domain: {
                        ...value.domain,
                        domain: {
                          ...value.domain.domain,
                          boundary: e.target.value,
                        },
                      },
                    })
                  }
                >
                  {["periodic", "dirichlet", "neumann", "mixed"].map((b) => (
                    <option key={b} value={b}>
                      {b === "periodic"
                        ? "periodic · opposite sides connect"
                        : `${b} · draft only, execution unavailable`}
                    </option>
                  ))}
                </select>
                <Textarea
                  aria-label="Boundary details"
                  value={value.boundary.details}
                  onChange={(e) =>
                    change({
                      ...value,
                      boundary: { ...value.boundary, details: e.target.value },
                    })
                  }
                />
              </CardContent>
            </Card>
            <Card id="initial-information">
              <CardHeader>
                <CardTitle>ℐ — Initial information</CardTitle>
              </CardHeader>
              <CardContent className="stack">
                <Status
                  label="Initial information"
                  value={value.initial.status}
                  onChange={(status) =>
                    change({ ...value, initial: { ...value.initial, status } })
                  }
                />
                <Textarea
                  aria-label="Initial details"
                  value={value.initial.details}
                  onChange={(e) =>
                    change({
                      ...value,
                      initial: { ...value.initial, details: e.target.value },
                    })
                  }
                />
                <p>
                  Static equations use not applicable. Wave initial displacement
                  and velocity require a future specialist. Notes do not impose
                  numeric constraints.
                </p>
              </CardContent>
            </Card>
            <Card>
              <CardHeader>
                <CardTitle>
                  y — Measurements; M — Operator and locations
                </CardTitle>
              </CardHeader>
              <CardContent className="stack">
                <p>
                  {value.measurements.rows.length} measurement rows. Edit/import
                  values and physical coordinates in Observations.
                </p>
                <Status
                  label="Measurement information status"
                  value={value.measurements.status}
                  onChange={(status) =>
                    change({
                      ...value,
                      measurements: { ...value.measurements, status },
                    })
                  }
                />
                <Status
                  label="Observation operator status"
                  value={value.observation_operator.status}
                  onChange={(status) =>
                    change({
                      ...value,
                      observation_operator: {
                        ...value.observation_operator,
                        status,
                      },
                    })
                  }
                />
                <select
                  aria-label="Observation operator"
                  value={value.observation_operator.kind}
                  onChange={(e) =>
                    change({
                      ...value,
                      observation_operator: {
                        ...value.observation_operator,
                        kind: e.target.value,
                      },
                    })
                  }
                >
                  <option value="point_on_grid">
                    Point values on saved nodes/times
                  </option>
                  <option value="off_grid">
                    Off-grid (execution unavailable)
                  </option>
                  <option value="integral">
                    Integral (execution unavailable)
                  </option>
                </select>
                <Textarea
                  aria-label="Observation operator details"
                  value={value.observation_operator.details}
                  onChange={(e) =>
                    change({
                      ...value,
                      observation_operator: {
                        ...value.observation_operator,
                        details: e.target.value,
                      },
                    })
                  }
                />
                <label>
                  Noise standard deviation
                  <Input
                    aria-label="Request noise standard deviation"
                    type="number"
                    value={value.measurements.noise_std}
                    onChange={(e) =>
                      change({
                        ...value,
                        measurements: {
                          ...value.measurements,
                          noise_std: Number(e.target.value),
                        },
                      })
                    }
                  />
                </label>
                <p>
                  Noise model: {value.measurements.noise_model}. Spatial
                  coordinate y is separate from the measurement vector y.
                  Locations are the row coordinates.
                </p>
              </CardContent>
            </Card>
            <Card>
              <CardHeader>
                <CardTitle>
                  <span id="model-override">Checkpoint compatibility</span>
                </CardTitle>
              </CardHeader>
              <CardContent className="stack">
                <select
                  aria-label="Registered checkpoint"
                  value={modelId}
                  onChange={(e) => {
                    onInvalid();
                    onModel(e.target.value);
                  }}
                >
                  <option value="">Automatic equation routing</option>
                  {models.map((m) => (
                    <option key={m.id} value={m.id}>
                      {m.name}
                    </option>
                  ))}
                </select>
                {report?.routing && (
                  <div aria-label="Routing report">
                    <p>
                      Recognized: {report.recognized_family} · selected
                      specialist: {report.routing.selected_specialist || "none"}
                    </p>
                    <p>
                      Canonical equation: {report.canonical_source.join("; ")}
                    </p>
                    <p>{report.routing.selection_reason}</p>
                    <p>
                      Compatible checkpoints:{" "}
                      {report.routing.compatible_checkpoints.length} · selected:{" "}
                      {report.routing.selected_checkpoint || "none"}
                    </p>
                    {report.routing.unknown_ranges.map((r) => (
                      <p key={r}>{r}</p>
                    ))}
                    {report.routing.range_warnings.map((r) => (
                      <p key={r.id}>{r.reason}</p>
                    ))}
                    <details>
                      <summary>Checkpoint exclusions and restrictions</summary>
                      {report.routing.exclusions.map((r) => (
                        <div key={r.id}>
                          {r.id}: {r.reason}
                          {r.compatibility && (
                            <pre>
                              {JSON.stringify(
                                r.compatibility.mismatches,
                                null,
                                2,
                              )}
                            </pre>
                          )}
                        </div>
                      ))}
                    </details>
                  </div>
                )}
                <p>{model?.training.status ?? "No checkpoint selected"}</p>
                <p>
                  Recorded D range:{" "}
                  {model?.training.diffusivity_range?.join(" – ") ?? "unknown"}
                </p>
                <Button
                  variant="outline"
                  disabled={!model}
                  onClick={() =>
                    model && propose(checkpointLayout(value, model))
                  }
                >
                  Use checkpoint layout
                </Button>
                <Button
                  disabled={advancedInvalid}
                  onClick={() => {
                    const version = ++sequence.current;
                    onInvalid();
                    setParsing(true);
                    void api<ParseReport>("/mathematics/parse", {
                      request: value,
                      model_id: modelId || null,
                    })
                      .then((r) => {
                        if (version !== sequence.current) return;
                        setReport(r);
                        onReport?.(r);
                        setError("");
                        onParsed(r.problem);
                        onRoute?.(r.routing?.selected_checkpoint ?? "");
                      })
                      .catch((e) => {
                        if (version !== sequence.current) return;
                        setError(String(e));
                        onParsed(null);
                      })
                      .finally(() => {
                        if (version === sequence.current) setParsing(false);
                      });
                  }}
                >
                  Validate problem
                </Button>
              </CardContent>
            </Card>
          </div>
        </div>
      </details>
      <details className="expert-export">
        <summary>Advanced training configuration (CLI)</summary>
        <TrainingDataCard model={model} />
      </details>
      <details className="expert-export" open={advancedInvalid || undefined}>
        <summary>Request JSON, import & export</summary>
        <Card>
          <CardHeader>
            <CardTitle>Live complete request and versioned source</CardTitle>
            <CardDescription>
              All 𝒞 components are exported, including unsupported or incomplete
              mathematics. Parse output and executable ProblemSpec are separate
              read-only views.
            </CardDescription>
          </CardHeader>
          <CardContent className="stack">
            <p aria-live="polite">
              𝓔: {value.equations.join("; ") || "empty"} · λ:{" "}
              {value.declarations
                .filter((d) => d.kind !== "field")
                .map((d) => `${d.name}=${d.value ?? d.status}`)
                .join(", ")}{" "}
              · Ω: {dimension}D {value.domain.shape.join("×")}, τ=[
              {value.domain.times.join(", ")}] · ℬ: {value.boundary.type}/
              {value.boundary.status} · ℐ: {value.initial.status} · y:{" "}
              {value.measurements.rows.length} rows, σ=
              {value.measurements.noise_std} · M:{" "}
              {value.observation_operator.kind}
            </p>
            <Textarea
              aria-label="Complete mathematical request JSON"
              className="code-editor"
              value={advanced ?? JSON.stringify(value, null, 2)}
              onChange={(e) => {
                const text = e.target.value;
                setAdvanced(text);
                sequence.current++;
                onInvalid();
                setReport(null);
                try {
                  const imported = importMathematics(text);
                  onChange(imported);
                  setAdvancedInvalid(false);
                  setError("");
                } catch (err) {
                  setAdvancedInvalid(true);
                  setError(String(err));
                }
              }}
            />
            <div className="button-row">
              <Button
                variant="outline"
                onClick={() =>
                  saveFile("mathematical-request.json", advanced ?? value)
                }
              >
                Export complete request
              </Button>
              <label className="file-input-label">
                Import complete request
                <Input
                  aria-label="Import complete request"
                  type="file"
                  accept=".json"
                  onChange={(e) => {
                    const file = e.target.files?.[0];
                    if (file)
                      void file.text().then((text) => {
                        try {
                          change(importMathematics(text));
                        } catch (err) {
                          setAdvanced(text);
                          setAdvancedInvalid(true);
                          sequence.current++;
                          onInvalid();
                          setError(String(err));
                        }
                      });
                  }}
                />
              </label>
            </div>
            <details>
              <summary>Parsed equation representation</summary>
              <pre>{JSON.stringify(report?.parsed ?? null, null, 2)}</pre>
            </details>
            <details>
              <summary>Executable ProblemSpec</summary>
              <pre>{JSON.stringify(report?.problem ?? null, null, 2)}</pre>
            </details>
          </CardContent>
        </Card>
      </details>
      <Dialog
        open={pending !== null}
        onOpenChange={(open) => !open && setPending(null)}
      >
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Review mathematical request changes</DialogTitle>
            <DialogDescription>
              Measurements and boundary information are preserved. Matching
              declaration values are retained. Review removed declarations,
              equation and time/initial-status changes before applying.
            </DialogDescription>
          </DialogHeader>
          <div className="preset-diff">
            <strong>Before</strong>
            <pre>
              {JSON.stringify(
                {
                  equations: value.equations,
                  declarations: value.declarations,
                  domain: value.domain,
                  boundary: value.boundary,
                  initial: value.initial,
                },
                null,
                2,
              )}
            </pre>
            <strong>After</strong>
            <pre>
              {JSON.stringify(
                pending && {
                  equations: pending.equations,
                  declarations: pending.declarations,
                  domain: pending.domain,
                  boundary: pending.boundary,
                  initial: pending.initial,
                },
                null,
                2,
              )}
            </pre>
          </div>
          <Button
            onClick={() => {
              if (pending) change(pending);
              setPending(null);
            }}
          >
            Apply reviewed changes
          </Button>
          <Button variant="outline" onClick={() => setPending(null)}>
            Keep current draft
          </Button>
        </DialogContent>
      </Dialog>
    </div>
  );
}
