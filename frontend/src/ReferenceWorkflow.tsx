import { AnalyticalExamples } from "./AnalyticalExamples";
import { useEffect, useState } from "react";
import { api } from "./types";
import type { Problem, Row, Slice } from "./types";
import type { MathematicalRequest } from "./mathematics";
import { FieldPlot } from "./FieldPlot";
import { Button } from "./components/ui/button";
import {
  Card,
  CardHeader,
  CardTitle,
  CardDescription,
  CardContent,
} from "./components/ui/card";
type Reference = {
  id: string;
  problem: Problem;
  diagnostics: Record<string, unknown>;
};
type Sampling = {
  sampling_id: string;
  indices: number[][];
  rows: Row[];
  mathematical_request: MathematicalRequest;
  count: number;
  disclosure: Record<string, string>;
};
function physicalKey(math: MathematicalRequest) {
  const { measurements: _measurements, preset: _preset, ...rest } = math;
  void _measurements;
  void _preset;
  return JSON.stringify(rest);
}
function readSaved() {
  try {
    return JSON.parse(localStorage.getItem("pdeflow.reference.v1") || "null");
  } catch {
    return null;
  }
}
export function ReferenceWorkflow({
  math,
  problem,
  ready,
  visible,
  onSample,
  onReconstruct,
}: {
  math: MathematicalRequest;
  problem: Problem;
  ready: boolean;
  visible: boolean;
  onSample: (v: Sampling, reference: string | null) => void;
  onReconstruct: () => void;
}) {
  const [saved] = useState(readSaved);
  const [inputs, setInputs] = useState<string>(saved?.inputs ?? "{}"),
    [solver, setSolver] = useState<string>(saved?.solver ?? "auto"),
    [accuracy, setAccuracy] = useState(saved?.accuracy ?? "{}"),
    [reference, setReference] = useState<Reference | null>(
      saved?.reference ?? null,
    ),
    [solvedKey, setSolvedKey] = useState<string>(saved?.solvedKey ?? ""),
    [sampling, setSampling] = useState<Sampling | null>(
      saved?.sampling ?? null,
    ),
    [error, setError] = useState(""),
    [busy, setBusy] = useState(false),
    [preview, setPreview] = useState<Slice | null>(null),
    [truth, setTruth] = useState(true),
    [field, setField] = useState(0),
    [time, setTime] = useState(0),
    [axis, setAxis] = useState(2),
    [index, setIndex] = useState(0),
    [pattern, setPattern] = useState(saved?.pattern ?? "fixed_sensors"),
    [count, setCount] = useState<number>(saved?.count ?? 10),
    [seed, setSeed] = useState<number>(saved?.seed ?? 54),
    [noise, setNoise] = useState<number>(saved?.noise ?? 0.03),
    [selected, setSelected] = useState<string[]>(saved?.selected ?? []),
    [times, setTimes] = useState(saved?.times ?? ""),
    [positions, setPositions] = useState(saved?.positions ?? "[]"),
    [disclosure, setDisclosure] = useState<Record<string, string>>(
      saved?.disclosure ?? {},
    ),
    [inputField, setInputField] = useState("");
  const [sourceMode, setSourceMode] = useState(
      saved?.sourceMode ?? "numerical",
    ),
    [referenceShape, setReferenceShape] = useState(saved?.referenceShape ?? ""),
    [referenceTimes, setReferenceTimes] = useState(saved?.referenceTimes ?? ""),
    [importField, setImportField] = useState(""),
    [measurementText, setMeasurementText] = useState(
      saved?.measurementText ?? "[]",
    ),
    [noiseMode, setNoiseMode] = useState(saved?.noiseMode ?? "absolute");
  const currentKey =
    physicalKey(math) +
    inputs +
    solver +
    accuracy +
    referenceShape +
    referenceTimes;
  const scalarCount = ["space_time", "positions", "reuse"].includes(pattern)
    ? pattern === "positions"
      ? (() => {
          try {
            return JSON.parse(positions).length;
          } catch {
            return 0;
          }
        })()
      : pattern === "reuse"
        ? (sampling?.indices.length ?? 0)
        : count
    : count *
      (selected.length || problem.fields.length) *
      (times.trim() ? times.split(",").length : problem.grid.times.length);
  const stale = !!reference && currentKey !== solvedKey;
  useEffect(() => {
    try {
      localStorage.setItem(
        "pdeflow.reference.v1",
        JSON.stringify({
          inputs,
          solver,
          reference,
          solvedKey,
          sampling,
          sourceMode,
          accuracy,
          referenceShape,
          referenceTimes,
          pattern,
          count,
          seed,
          noise,
          selected,
          times,
          positions,
          disclosure,
          noiseMode,
          measurementText,
        }),
      );
    } catch {
      /* Main draft export remains available. */
    }
  }, [
    inputs,
    solver,
    reference,
    solvedKey,
    sampling,
    sourceMode,
    accuracy,
    referenceShape,
    referenceTimes,
    pattern,
    count,
    seed,
    noise,
    selected,
    times,
    positions,
    disclosure,
    noiseMode,
    measurementText,
  ]);
  async function act(fn: () => Promise<void>) {
    setBusy(true);
    setError("");
    try {
      await fn();
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  }
  function request() {
    const full = JSON.parse(inputs);
    const request = structuredClone(math);
    if (referenceShape.trim())
      request.domain.shape = referenceShape.split(",").map(Number);
    if (referenceTimes.trim())
      request.domain.times = referenceTimes.split(",").map(Number);
    if (full.forcing !== undefined) {
      request.known_fields = { ...request.known_fields, f: full.forcing };
      delete full.forcing;
    }
    return {
      request,
      inputs: full,
      solver,
      accuracy: JSON.parse(accuracy),
      seed: 53,
    };
  }
  const ref = reference?.problem;
  useEffect(() => {
    if (!visible || !reference || !truth) return;
    let live = true;
    void api<Slice>(
      `/references/${reference.id}/slice?field_index=${field}&time_index=${time}&axis=${axis}&index=${index}${sampling ? `&sampling_id=${sampling.sampling_id}` : ""}`,
    )
      .then((v) => {
        if (live) setPreview(v);
      })
      .catch((e) => {
        if (live) setError(String(e));
      });
    return () => {
      live = false;
    };
  }, [reference, field, time, axis, index, sampling, truth, visible]);
  function defaults() {
    const names = problem.fields.map((f) => f.name);
    const values =
      problem.family === "heat"
        ? {
            initial_u: {
              fourier: {
                offset: 0.2,
                modes: [
                  {
                    wave: problem.grid.shape.map(() => 1),
                    amplitude: 0.6,
                    phase: 0,
                  },
                ],
              },
            },
          }
        : problem.family === "darcy"
          ? {
              permeability: { expression: "exp(0.3*cos(2*pi*x))" },
              forcing: math.known_fields?.f ?? { expression: "sin(2*pi*x)" },
            }
          : {
              initial_a: { expression: "0.3+0.1*cos(2*pi*x)" },
              initial_b: { expression: "0.4+0.1*sin(2*pi*x)" },
              initial_c: 0.1,
            };
    setInputs(JSON.stringify(values, null, 2));
    setSelected(names);
    setDisclosure(
      Object.fromEntries(names.map((n) => [n, "partially_measured"])),
    );
    setInputField(Object.keys(values)[0]);
    setPreview(null);
  }
  return (
    <section hidden={!visible} className="reference-workflow">
      <Card>
        <CardHeader>
          <CardTitle>Reference experiment</CardTitle>
          <CardDescription>
            Complete solver inputs stay outside the inverse worker. Solve
            without a trained checkpoint; then disclose only measurements.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <label>
            Reference / measurement source
            <select
              aria-label="Input source mode"
              value={sourceMode}
              onChange={(e) => {
                setSourceMode(e.target.value);
                setError("");
                if (e.target.value === "analytical") setSolver("analytical");
                if (e.target.value === "numerical") setSolver("numerical");
              }}
            >
              <option value="analytical">
                Analytical / manufactured reference
              </option>
              <option value="numerical">Numerical reference</option>
              <option value="measurements">Measurements only</option>
            </select>
          </label>
          {sourceMode === "measurements" && (
            <>
              <p>
                No full reference is required. Enter/import scalar measurement
                rows (measured zeros are values; missing rows are unmeasured).
                Truth error is unavailable without an attached reference;
                supported physics diagnostics remain available. Existing
                fixed-grid specialists require exact grid nodes/times; the
                coordinate specialist also accepts scattered sensors in its own
                panel.
              </p>
              <label>
                Measurements JSON rows
                <textarea
                  aria-label="Measurements-only rows"
                  value={measurementText}
                  onChange={(e) => setMeasurementText(e.target.value)}
                />
              </label>
              <label>
                Import measurements JSON
                <input
                  type="file"
                  accept=".json"
                  aria-label="Measurements-only file"
                  onChange={(e) => {
                    const f = e.target.files?.[0];
                    if (f)
                      void act(async () => setMeasurementText(await f.text()));
                  }}
                />
              </label>
              <Button
                disabled={!ready || busy}
                onClick={() =>
                  void act(async () => {
                    const v = await api<{
                      rows: Row[];
                      count: number;
                      observations: { indices: number[][] };
                    }>("/observations/validate", {
                      problem,
                      rows: JSON.parse(measurementText),
                      noise_std: noise,
                    });
                    const m = structuredClone(math);
                    m.measurements = {
                      ...m.measurements,
                      rows: v.rows,
                      status: "known",
                    };
                    onSample(
                      {
                        sampling_id: "",
                        rows: v.rows,
                        count: v.count,
                        indices: v.observations.indices,
                        mathematical_request: m,
                        disclosure: {},
                      },
                      null,
                    );
                    setReference(null);
                    setSampling(null);
                  })
                }
              >
                Adopt measurements without reference
              </Button>
              <Button variant="outline" onClick={onReconstruct}>
                Open reconstruction
              </Button>
            </>
          )}
          <div hidden={sourceMode === "measurements"}>
            <p>
              Current equation: {problem.family} ·{" "}
              {problem.grid.shape.join(" × ")} ·{" "}
              {problem.grid.temporal_kind === "static"
                ? "static, zero-mean pressure gauge"
                : `saved physical times ${math.domain.times.join(", ")}`}{" "}
              · periodic boundary. Coefficients and domain are edited above.
            </p>
            <Button variant="outline" disabled={!ready} onClick={defaults}>
              Load visible reference defaults
            </Button>
            <div hidden={sourceMode !== "analytical"}>
              <AnalyticalExamples
                problem={problem}
                expanded={sourceMode === "analytical"}
                ready={ready}
                inputs={inputs}
                onInputs={(v) => {
                  setInputs(v);
                  setSelected(["u"]);
                  setDisclosure({ u: "partially_measured" });
                  setInputField("initial_u");
                  setPreview(null);
                }}
                onVerify={() => api("/analytical-verify", request())}
              />
            </div>
            <label>
              Complete reference inputs (JSON: scalar, array, expression or
              finite Fourier modes)
              <textarea
                aria-label="Complete reference inputs"
                rows={8}
                value={inputs}
                onChange={(e) => {
                  setInputs(e.target.value);
                  setPreview(null);
                }}
              />
            </label>
            <label>
              Import reference input JSON file
              <input
                aria-label="Reference input file"
                type="file"
                accept=".json"
                onChange={(e) => {
                  const f = e.target.files?.[0];
                  if (f)
                    void act(async () => {
                      const text = await f.text();
                      JSON.parse(text);
                      setInputs(text);
                    });
                }}
              />
            </label>
            <p>
              Heat requires initial_u; Darcy requires permeability (forcing
              optionally overrides f); chemistry requires initial_a, initial_b,
              initial_c. Expressions use physical x/y/z, pi,
              sin/cos/exp/sqrt/tanh/abs and bounded arithmetic. Fourier modes
              use domain-relative integer wave numbers. Full
              initials/permeability are reference-only.
            </p>
            {sourceMode === "numerical" && (
              <details>
                <summary>Import complete numerical reference JSON</summary>
                <p>
                  Format: problem, coordinates by axis, physical_times, ordered
                  fields, values[field,time,*space], units="nondimensional",
                  provenance. Uniform increasing coordinates must exactly match
                  the explicit grid; no inferred axes, snapping or
                  extrapolation. JSON upload is limited to2MB; CLI supports
                  larger files.
                </p>
                <textarea
                  aria-label="Numerical reference artifact"
                  value={importField}
                  onChange={(e) => setImportField(e.target.value)}
                />
                <input
                  type="file"
                  accept=".json"
                  aria-label="Numerical reference file"
                  onChange={(e) => {
                    const f = e.target.files?.[0];
                    if (f) void act(async () => setImportField(await f.text()));
                  }}
                />
                <Button
                  disabled={busy || !ready}
                  onClick={() =>
                    void act(async () => {
                      const artifact = JSON.parse(importField);
                      if (artifact.problem.family !== problem.family)
                        throw Error(
                          "Imported family differs from the selected equation. Select and review the intended equation first.",
                        );
                      const v = await api<Reference>("/references/import", {
                        artifact,
                      });
                      setReference(v);
                      setSolvedKey(currentKey);
                      setSampling(null);
                      setField(0);
                      setTime(0);
                      setIndex(0);
                    })
                  }
                >
                  Import complete reference field
                </Button>
              </details>
            )}
            <div className="preview-controls">
              <label>
                Numerical/reference grid nodes (separate from model/query grid)
                <input
                  aria-label="Reference grid nodes"
                  placeholder={problem.grid.shape.join(",")}
                  value={referenceShape}
                  onChange={(e) => setReferenceShape(e.target.value)}
                />
              </label>
              <label>
                Reference physical output times
                <input
                  aria-label="Reference output times"
                  placeholder={problem.grid.times.join(",")}
                  value={referenceTimes}
                  onChange={(e) => setReferenceTimes(e.target.value)}
                />
              </label>
            </div>
            <p>
              Preview plotting is bounded to128 points per visible axis; it does
              not change solver resolution, observation count or model query
              resolution. Known coefficients/domain/boundary are explicit above;
              initial inputs below are reference-only.
            </p>
            <div className="preview-controls">
              <label>
                Reference solver
                <select
                  aria-label="Reference solver"
                  value={solver}
                  onChange={(e) => setSolver(e.target.value)}
                >
                  {["auto", "analytical", "numerical", "manufactured"].map(
                    (v) => (
                      <option key={v}>{v}</option>
                    ),
                  )}
                </select>
              </label>
              <label>
                Accuracy settings (JSON)
                <input
                  aria-label="Reference accuracy"
                  value={accuracy}
                  onChange={(e) => setAccuracy(e.target.value)}
                />
              </label>
              <label>
                Input to preview
                <input
                  aria-label="Reference input field"
                  value={inputField}
                  onChange={(e) => setInputField(e.target.value)}
                />
              </label>
            </div>
            <div className="toolbar">
              <Button
                variant="outline"
                disabled={!ready || busy}
                onClick={() =>
                  void act(async () =>
                    setPreview(
                      await api<Slice>("/reference-input-preview", {
                        ...request(),
                        field: inputField === "forcing" ? "f" : inputField,
                        axis,
                        index,
                      }),
                    ),
                  )
                }
              >
                Preview supplied field
              </Button>
              <Button
                disabled={!ready || busy}
                onClick={() =>
                  void act(async () => {
                    const v = await api<Reference>("/references", request());
                    setReference(v);
                    setSolvedKey(currentKey);
                    setSampling(null);
                    setField(0);
                    setTime(0);
                    setIndex(0);
                  })
                }
              >
                Solve reference
              </Button>
            </div>
            {stale && (
              <p role="alert">
                Reference is stale: complete physical inputs changed. Solve a
                new reference before sampling.
              </p>
            )}
            {reference && (
              <>
                <p>
                  Immutable reference {reference.id} ·{" "}
                  {String(reference.diagnostics.classification)}
                </p>
                <details>
                  <summary>Solver provenance and numerical diagnostics</summary>
                  <pre>{JSON.stringify(reference.diagnostics, null, 2)}</pre>
                </details>
                <a
                  href={`/api/references/${reference.id}/artifacts/reference.npz`}
                >
                  Export reference arrays
                </a>
                {" · "}
                <a
                  href={`/api/references/${reference.id}/artifacts/reference-problem.json`}
                >
                  Export complete solver inputs
                </a>
              </>
            )}
            <label>
              <input
                type="checkbox"
                checked={truth}
                onChange={(e) => setTruth(e.target.checked)}
              />{" "}
              Show reference truth (display only; inverse inputs unchanged)
            </label>
            {ref && (
              <div className="preview-controls">
                <label>
                  Reference field
                  <select
                    aria-label="Reference field"
                    value={field}
                    onChange={(e) => setField(Number(e.target.value))}
                  >
                    {ref.fields.map((f, i) => (
                      <option key={f.name} value={i}>
                        {f.name}
                      </option>
                    ))}
                  </select>
                </label>
                {ref.grid.temporal_kind !== "static" && (
                  <label>
                    Reference physical time
                    <select
                      aria-label="Reference physical time"
                      value={time}
                      onChange={(e) => setTime(Number(e.target.value))}
                    >
                      {ref.grid.times.map((t, i) => (
                        <option key={i} value={i}>
                          {t}
                        </option>
                      ))}
                    </select>
                  </label>
                )}
                {ref.grid.shape.length === 3 && (
                  <>
                    <label>
                      Reference slice normal
                      <select
                        aria-label="Reference slice normal"
                        value={axis}
                        onChange={(e) => {
                          setAxis(Number(e.target.value));
                          setIndex(0);
                        }}
                      >
                        {["x", "y", "z"].map((a, i) => (
                          <option key={a} value={i}>
                            {a}
                          </option>
                        ))}
                      </select>
                    </label>
                    <label>
                      Reference slice index
                      <input
                        aria-label="Reference slice index"
                        type="range"
                        min={0}
                        max={ref.grid.shape[axis] - 1}
                        value={index}
                        onChange={(e) => setIndex(Number(e.target.value))}
                      />
                    </label>
                  </>
                )}
              </div>
            )}
            {truth && preview && (
              <>
                <p>
                  {preview.observations.length} displayed observations in this
                  physical-time/slice;{" "}
                  {preview.off_slice_observation_count ?? 0} off this view.
                </p>
                <FieldPlot
                  slice={preview}
                  values={preview.values}
                  title={
                    reference
                      ? "Reference field preview"
                      : "Supplied reference input"
                  }
                  observations
                />
              </>
            )}
            {!truth && (
              <p>
                Truth hidden. No reference values are added to conditioning;
                noisy measurements remain visible in the observation table.
              </p>
            )}
            <h3>Observation generation</h3>
            <p>
              Preview total: {scalarCount} scalar observations. Independent
              Gaussian noise only; random discrete nodes are selected without
              replacement.
            </p>
            <details open={!!reference}>
              <summary>What is known in the reconstruction?</summary>
              <p>
                Parameters and forcing are fully known. Full initial states are
                hidden. Current joint specialists accept sparse target
                measurements, not a fully known target field.
              </p>
              <table aria-label="Reference disclosure">
                <thead>
                  <tr>
                    <th>Quantity</th>
                    <th>Inverse information</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.keys(problem.known_parameters)
                    .concat(Object.keys(problem.known_fields ?? {}))
                    .map((n) => (
                      <tr key={n}>
                        <td>{n}</td>
                        <td>Fully known</td>
                      </tr>
                    ))}
                  {problem.fields.map((f) => (
                    <tr key={f.name}>
                      <td>{f.name}</td>
                      <td>
                        <select
                          aria-label={`Disclosure ${f.name}`}
                          value={disclosure[f.name] ?? "partially_measured"}
                          onChange={(e) =>
                            setDisclosure((v) => ({
                              ...v,
                              [f.name]: e.target.value,
                            }))
                          }
                        >
                          {["partially_measured", "unknown", "fully_known"].map(
                            (s) => (
                              <option key={s}>{s}</option>
                            ),
                          )}
                        </select>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
              <label>
                Sampling pattern
                <select
                  aria-label="Sampling pattern"
                  value={pattern}
                  onChange={(e) => setPattern(e.target.value)}
                >
                  {[
                    "fixed_sensors",
                    "random_spatial",
                    "regular",
                    "space_time",
                    "positions",
                    "reuse",
                  ].map((p) => (
                    <option key={p}>{p}</option>
                  ))}
                </select>
              </label>
              <div className="preview-controls">
                <label>
                  Count
                  <input
                    aria-label="Sampling count"
                    type="number"
                    min={1}
                    value={count}
                    onChange={(e) => setCount(Number(e.target.value))}
                  />
                </label>
                <label>
                  Sampling seed
                  <input
                    aria-label="Sampling seed"
                    type="number"
                    value={seed}
                    onChange={(e) => setSeed(Number(e.target.value))}
                  />
                </label>
                <label>
                  Gaussian noise magnitude
                  <input
                    aria-label="Sampling noise"
                    type="number"
                    min={0}
                    step={0.01}
                    value={noise}
                    onChange={(e) => setNoise(Number(e.target.value))}
                  />
                </label>
              </div>
              <label>
                Noise interpretation
                <select
                  aria-label="Sampling noise mode"
                  value={noiseMode}
                  onChange={(e) => setNoiseMode(e.target.value)}
                >
                  <option value="absolute">
                    Absolute standard deviation σ
                  </option>
                  <option value="relative_rms">
                    Relative to full reference field/time RMS
                  </option>
                </select>
              </label>
              <p>
                Relative noise magnitude times the full selected field RMS at
                each physical time gives absolute sigma (zero RMS gives sigma0).
                Actual sigma is saved per measurement. No correlated noise.
              </p>
              <p>
                Fixed/regular/random spatial: count per selected field and saved
                time. Space-time: total count. Positions/reuse: explicit pattern
                length. Exact grid coordinates and saved times only; no
                snapping/interpolation.
              </p>
              <fieldset>
                <legend>Measured fields</legend>
                {problem.fields.map((f) => (
                  <label key={f.name}>
                    <input
                      type="checkbox"
                      checked={selected.includes(f.name)}
                      onChange={(e) =>
                        setSelected((v) =>
                          e.target.checked
                            ? [...v, f.name]
                            : v.filter((n) => n !== f.name),
                        )
                      }
                    />
                    {f.name}
                  </label>
                ))}
              </fieldset>
              <label>
                Time indices (comma separated; blank = all saved times)
                <input
                  aria-label="Sampling time indices"
                  value={times}
                  onChange={(e) => setTimes(e.target.value)}
                />
              </label>
              {pattern === "positions" && (
                <label>
                  Exact positions JSON rows
                  <textarea
                    aria-label="Sampling positions"
                    value={positions}
                    onChange={(e) => setPositions(e.target.value)}
                  />
                </label>
              )}
              <Button variant="outline" onClick={() => setSeed((v) => v + 1)}>
                Resample: advance observation seed
              </Button>
              <div className="toolbar">
                <Button
                  disabled={!reference || stale || busy}
                  onClick={() =>
                    void act(async () => {
                      const v = await api<Sampling>(
                        `/references/${reference!.id}/sample`,
                        {
                          sampling: {
                            pattern,
                            count,
                            seed,
                            noise_std: noise,
                            noise_mode: noiseMode,
                            fields: selected,
                            time_indices: times.trim()
                              ? times.split(",").map(Number)
                              : [],
                            disclosure,
                            positions: JSON.parse(positions),
                            reuse_indices:
                              pattern === "reuse"
                                ? (sampling?.indices ?? [])
                                : [],
                          },
                        },
                      );
                      setSampling(v);
                      setSolvedKey(
                        physicalKey(v.mathematical_request) +
                          inputs +
                          solver +
                          accuracy +
                          referenceShape +
                          referenceTimes,
                      );
                      onSample(v, reference!.id);
                    })
                  }
                >
                  Generate observations
                </Button>
                <Button
                  variant="outline"
                  disabled={!sampling || stale}
                  onClick={onReconstruct}
                >
                  Reconstruct from measurements
                </Button>
              </div>
              {sampling && (
                <>
                  <p>
                    {sampling.count} noisy measurements. Clean sampled values
                    are retained separately for evaluation only.
                  </p>
                  <details>
                    <summary>Observed values and absolute noise</summary>
                    <table aria-label="Generated observation values">
                      <thead>
                        <tr>
                          <th>Field</th>
                          <th>Physical coordinates/time</th>
                          <th>Observed (noisy) value</th>
                          <th>Absolute σ</th>
                        </tr>
                      </thead>
                      <tbody>
                        {sampling.rows.slice(0, 200).map((r, i) => (
                          <tr key={i}>
                            <td>{r.field}</td>
                            <td>
                              {problem.grid.shape
                                .map(
                                  (_, j) =>
                                    `${["x", "y", "z"][j]}=${r[["x", "y", "z"][j]]}`,
                                )
                                .join(", ")}
                              ; t={r.time ?? "static"}
                            </td>
                            <td>{r.value}</td>
                            <td>{r.sigma ?? noise}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                    <p>
                      Showing first200 of{sampling.rows.length}; full rows
                      remain exportable.
                    </p>
                  </details>
                  <table aria-label="Measurement coverage">
                    <thead>
                      <tr>
                        <th>Field / time</th>
                        <th>Count</th>
                      </tr>
                    </thead>
                    <tbody>
                      {Object.entries(
                        sampling.rows.reduce<Record<string, number>>((a, r) => {
                          const k = `${r.field} / ${r.time ?? "static"}`;
                          a[k] = (a[k] ?? 0) + 1;
                          return a;
                        }, {}),
                      ).map(([key, n]) => (
                        <tr key={key}>
                          <td>{key}</td>
                          <td>{n}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                  <a
                    href={`/api/references/${reference!.id}/artifacts/observations.csv?sampling_id=${sampling.sampling_id}`}
                  >
                    Export measured CSV
                  </a>
                  {" · "}
                  <a
                    href={`/api/references/${reference!.id}/artifacts/inverse-request.json?sampling_id=${sampling.sampling_id}`}
                  >
                    Export disclosed inverse request
                  </a>
                </>
              )}
            </details>
          </div>
          {busy && <p role="status">Computing reference experiment…</p>}
          {error && <p role="alert">{error}</p>}
        </CardContent>
      </Card>
    </section>
  );
}
