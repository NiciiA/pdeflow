import { useEffect, useMemo, useRef, useState } from "react";
import { api } from "./types";
import type { Problem, Slice } from "./types";
import type { MathematicalRequest } from "./mathematics";
import { Button } from "./components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "./components/ui/card";
import { FieldPlot } from "./FieldPlot";
import { BatchControls, type BatchStage } from "./BatchControls";
import { FieldLossControls } from "./FieldLossControls";
import { LineChart } from "./ScienceCharts";

export type TrainingSelection = {
  id: string;
  model: string;
  suggested_request: MathematicalRequest;
  adopt_training_problem: boolean;
  selected_at?: string;
};
type Options = {
  problems: string[];
  models: Record<string, string>;
  budgets: Record<string, string>;
  devices: Record<string, boolean>;
  auto_device: string;
  auto_explanation: string;
};
type Plan = {
  batch_settings?: Record<string, BatchStage>;
  problem: string;
  model: string;
  model_label: string;
  equation: string;
  varies: string;
  training_physics?: {
    family_weights: Record<string, number>;
    state_schemas: Record<string, string[]>;
    equations: Record<string, string>;
    parameter_ranges: Record<string, number[]>;
    feature_mixture_meaning: string;
  };
  resolved_device: string;
  grid: Problem["grid"];
  updates: Record<string, number>;
  examples: Record<string, number | string>;
  distribution: Record<string, unknown>;
  config: Record<string, unknown>;
  budget_meaning: string;
  architecture_parameters?: {
    encoder: number;
    decoder: number;
    observation: number;
    context: number;
    velocity: number;
    inference: number;
    total_training: number;
  };
  pipeline_check: boolean;
  stages: string[];
};
type History = {
  step: number;
  training_loss: number;
  validation_loss?: number;
};
type Run = {
  id: string;
  state: string;
  stage: string;
  stage_label: string;
  stage_meaning: string;
  progress: number;
  stage_budget: number;
  elapsed_seconds: number;
  eta_seconds: number | null;
  eta_scope: string;
  history: History[];
  finished_stages: string[];
  plan: Plan;
  validation: unknown;
  representation_validation?: unknown;
  representation_quality?: {
    passed: boolean;
    validation_mse: number;
    constant_baseline_mse: number;
    mse_ratio: number | null;
  };
  error?: string;
  warning?: string;
  model_id: string | null;
  can_resume: boolean;
  stop_requested: boolean;
  log_tail: string;
  stage_log_tail: string;
  training_examples_consumed?: number | null;
  meaning: string;
};
type Model = {
  id: string;
  model: string;
  type: string;
  problem: Problem;
  status: string;
  updates: Record<string, number>;
  validation: unknown;
  quality: string;
  pipeline_check: boolean;
  workflow_id: string | null;
  can_extend?: boolean;
  compatibility: { compatible: boolean; summary: string };
};
const stageNames: Record<string, string> = {
  prepare: "Prepare examples",
  representation: "A1 · Learn state representation",
  evolution: "A2 · Learn physical evolution",
  representation_validation: "Check representation",
  flow: "B · Learn conditional generation",
  fixed: "Learn fixed-grid generation",
  model_validation: "Inspect validation reconstructions",
  register: "Make model available",
};
const deviceNames: Record<string, string> = {
  auto: "Auto",
  cpu: "CPU",
  mps: "Apple GPU",
  cuda: "NVIDIA GPU",
};
const budgetNames: Record<string, string> = {
  "pipeline-check": "Pipeline check",
  development: "Development run",
  extended: "Extended run",
};

function pendingToken() {
  try {
    return (
      localStorage.getItem("pdeflow.training.pending-token") ||
      crypto.randomUUID()
    );
  } catch {
    return crypto.randomUUID();
  }
}

function validationSummary(value: unknown): string[] {
  if (!value || typeof value !== "object") return [];
  const report = value as {
    records?: { ensemble_mean_rmse: number }[];
    reports?: Record<
      string,
      Record<
        string,
        { relative_l2_unobserved_mean?: number; observation_rmse?: number }
      >
    >;
  };
  if (report.records)
    return report.records.flatMap((value, i) => {
      const r = value as {
        family?: string;
        parameter_status?: string;
        quantities?: Record<string, { ensemble_mean_rmse?: number }>;
        ensemble_mean_rmse?: number;
      };
      if (r.quantities)
        return Object.entries(r.quantities).map(
          ([q, m]) =>
            `${r.family || "Validation"} ${r.parameter_status || ""} ${q}: physical ensemble-mean RMSE ${m.ensemble_mean_rmse?.toPrecision(4) ?? "unavailable"}.`,
        );
      return [
        `Validation case ${i + 1}: ensemble-mean physical RMSE ${r.ensemble_mean_rmse?.toPrecision(4) ?? "unavailable"}.`,
      ];
    });
  return Object.entries(report.reports || {}).map(([id, modes]) => {
    const m = modes.observation_pde;
    return `Validation case ${id}: unobserved relative L2 ${m?.relative_l2_unobserved_mean?.toPrecision(4) ?? "unavailable"}; noisy-observation RMSE ${m?.observation_rmse?.toPrecision(4) ?? "unavailable"}.`;
  });
}

function parameterRanges(
  distribution: Record<string, unknown>,
  problem: string,
) {
  const values = {
    ...distribution,
    ...((distribution.chemistry as Record<string, unknown>) || {}),
  };
  return Object.entries(values)
    .filter(
      ([key, value]) =>
        Array.isArray(value) &&
        (key.endsWith("_range") ||
          ["diffusivity", "wave_speed"].includes(key)) &&
        !(problem === "heat-2d" && key.startsWith("wave_speed")) &&
        !(problem === "wave-2d" && key.startsWith("diffusivity")),
    )
    .map(
      ([key, value]) =>
        `${key.replaceAll("_", " ")}: ${(value as number[]).join("–")}`,
    )
    .join(" · ");
}

export function TrainModel({
  problem,
  onUse,
  onRefresh,
  focused = false,
}: {
  problem?: Problem;
  onUse?: (selection: TrainingSelection) => void;
  onRefresh: () => void;
  focused?: boolean;
}) {
  const [options, setOptions] = useState<Options | null>(null);
  // New training intent is independent of the reference currently displayed.
  const [specialist, setSpecialist] = useState("heat-wave-2d");
  const [model, setModel] = useState("resolution-flexible");
  const [budget, setBudget] = useState(
    focused ? "development" : "pipeline-check",
  );
  const [device, setDevice] = useState("auto");
  const [advanced, setAdvanced] = useState<Record<string, unknown>>({});
  const [allowPoor, setAllowPoor] = useState(false);
  const [continuePoor, setContinuePoor] = useState(false);
  const [gateRatio, setGateRatio] = useState(1);
  const [plan, setPlan] = useState<Plan | null>(null);
  const [resolvedRequestKey, setResolvedRequestKey] = useState("");
  const [planValid, setPlanValid] = useState(false);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [batchDirty, setBatchDirty] = useState(false);
  const [batchBenchmarkRunning, setBatchBenchmarkRunning] = useState(false);
  const [runs, setRuns] = useState<Run[]>([]);
  const [runsLoaded, setRunsLoaded] = useState(false);
  const [selected, setSelected] = useState(
    () => localStorage.getItem("pdeflow.training.run") || "",
  );
  const [models, setModels] = useState<Model[]>([]);
  const [modelsLoaded, setModelsLoaded] = useState(false);
  const [loadedPreview, setLoadedPreview] = useState<{
    key: string;
    slice: Slice;
  } | null>(null);
  const [field, setField] = useState(0),
    [physicalTime, setPhysicalTime] = useState(0);
  const [axis, setAxis] = useState(2),
    [index, setIndex] = useState(0);
  const [revision, setRevision] = useState(0);
  const [additionalUpdates, setAdditionalUpdates] = useState(100);
  const inFlight = useRef(false),
    token = useRef(pendingToken());
  const refreshEpoch = useRef(0);
  const request = useMemo(
    () => ({
      problem: specialist,
      model,
      budget,
      device,
      advanced: {
        ...advanced,
        field_loss: {
          type: "mse",
          ...((advanced.field_loss as Record<string, unknown>) || {}),
        },
      },
      allow_poor_representation: allowPoor,
      representation_max_mse_ratio: gateRatio,
    }),
    [specialist, model, budget, device, advanced, allowPoor, gateRatio],
  );
  const requestKey = JSON.stringify(request);
  const planPending = requestKey !== resolvedRequestKey;
  useEffect(() => {
    void api<Options>("/training/options")
      .then(setOptions)
      .catch((e) => setError(String(e)));
  }, []);
  useEffect(() => {
    let live = true;
    const timer = setTimeout(
      () =>
        void api<Plan>("/training/plan", request)
          .then((p) => {
            if (live) {
              setPlan(p);
              setError("");
              setResolvedRequestKey(JSON.stringify(request));
              setPlanValid(true);
            }
          })
          .catch((e) => {
            if (live) {
              setError(String(e));
              setResolvedRequestKey(JSON.stringify(request));
              setPlanValid(false);
            }
          }),
      180,
    );
    return () => {
      live = false;
      clearTimeout(timer);
    };
  }, [request]);
  useEffect(() => {
    let live = true,
      fetching = false;
    const refresh = () => {
      if (fetching) return;
      fetching = true;
      const epoch = refreshEpoch.current;
      void api<Run[]>("/training/runs")
        .then((rows) => {
          if (!live || epoch !== refreshEpoch.current || inFlight.current)
            return;
          const visible = focused
            ? rows.filter(
                (r) =>
                  ["heat-2d", "wave-2d", "heat-wave-2d"].includes(
                    r.plan.problem,
                  ) && r.plan.model === "resolution-flexible",
              )
            : rows;
          setRuns(visible);
          setRunsLoaded(true);
          setSelected((old) =>
            visible.some((r) => r.id === old) ? old : visible[0]?.id || "",
          );
        })
        .catch((e) => {
          if (live) setError(String(e));
        })
        .finally(() => {
          fetching = false;
        });
    };
    refresh();
    const timer = setInterval(refresh, 1000);
    return () => {
      live = false;
      clearInterval(timer);
    };
  }, [focused]);
  const run = runs.find((r) => r.id === selected);
  const completion = runs.map((r) => `${r.id}:${r.state}`).join("|");
  const context = problem && problem.family !== "custom" ? problem : null;
  const contextKey = JSON.stringify(context);
  useEffect(() => {
    let live = true;
    void api<Model[]>("/training/models", { problem: context })
      .then((rows) => {
        if (live) {
          setModels(
            focused
              ? rows.filter(
                  (r) =>
                    r.model === "resolution-flexible" &&
                    r.problem.grid.shape.length === 2,
                )
              : rows,
          );
          setModelsLoaded(true);
        }
      })
      .catch((e) => {
        if (live) setError(String(e));
      });
    return () => {
      live = false;
    };
    // The serialized physical contract keeps object identity from causing polling.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [completion, contextKey, revision, focused]);
  useEffect(() => {
    if (selected) localStorage.setItem("pdeflow.training.run", selected);
  }, [selected]);
  const previewKey = `${run?.id}:${run?.state}:${field}:${physicalTime}:${axis}:${index}`;
  const slice = loadedPreview?.key === previewKey ? loadedPreview.slice : null;
  useEffect(() => {
    let live = true;
    if (run?.state === "completed")
      void api<Slice>(
        `/training/runs/${run.id}/preview?field=${field}&time_index=${physicalTime}&axis=${axis}&index=${index}`,
      )
        .then((s) => {
          if (live) setLoadedPreview({ key: previewKey, slice: s });
        })
        .catch((e) => {
          if (live) setError(String(e));
        });
    return () => {
      live = false;
    };
  }, [run?.id, run?.state, field, physicalTime, axis, index, previewKey]);
  function chooseRun(id: string) {
    setSelected(id);
    setField(0);
    setPhysicalTime(0);
    setIndex(0);
    setLoadedPreview(null);
    setContinuePoor(false);
  }
  function resetAdvanced() {
    setAdvanced({});
    setAllowPoor(false);
  }
  async function action(fn: () => Promise<void>) {
    if (inFlight.current) return;
    inFlight.current = true;
    refreshEpoch.current++;
    setBusy(true);
    setError("");
    try {
      await fn();
    } catch (e) {
      setError(String(e));
    } finally {
      inFlight.current = false;
      refreshEpoch.current++;
      setBusy(false);
    }
  }
  function setSetting(path: string, value: unknown) {
    const [section, key] = path.split(".");
    setAdvanced((old) =>
      key
        ? {
            ...old,
            [section]: { ...((old[section] as object) || {}), [key]: value },
          }
        : { ...old, [section]: value },
    );
  }
  function current(path: string) {
    const [section, key] = path.split(".");
    return key
      ? ((advanced[section] as Record<string, unknown>)?.[key] ??
          (plan?.config[section] as Record<string, unknown>)?.[key])
      : (advanced[section] ?? plan?.config[section]);
  }
  function number(label: string, path: string, step = 1, min = 1) {
    return (
      <label key={path}>
        {label}
        <input
          aria-label={label}
          type="number"
          step={step < 1 ? "any" : step}
          min={min}
          value={current(path) == null ? "" : Number(current(path))}
          onChange={(e) => setSetting(path, e.target.valueAsNumber)}
        />
      </label>
    );
  }
  function numbers(label: string, path: string) {
    return (
      <label key={path}>
        {label}
        <input
          aria-label={label}
          defaultValue={(current(path) as number[] | undefined)?.join(", ")}
          key={`${specialist}-${model}-${budget}-${path}`}
          onBlur={(e) =>
            setSetting(path, e.target.value.split(",").map(Number))
          }
        />
        <small>Comma-separated values; applied when leaving the field.</small>
      </label>
    );
  }
  const active = runs.some((r) =>
    ["queued", "running", "stopping"].includes(r.state),
  );
  const ownModel = models.find((m) => m.id === run?.model_id);
  async function use(m: Model) {
    const selection = await api<TrainingSelection>("/training/use", {
      id: m.id,
      problem: context,
      adopt_training_problem: !m.compatibility.compatible,
    });
    onRefresh();
    onUse?.(selection);
  }
  return (
    <section className="integrated-training">
      <div className="section-intro">
        <h1>Train model</h1>
        <p>
          Choose training equations, model capacity and budget. The application
          prepares data, manages stages and makes the result available.
        </p>
      </div>
      <Card>
        <CardHeader>
          <CardTitle>Train a PDE model</CardTitle>
        </CardHeader>
        <CardContent>
          <form
            noValidate
            onSubmit={(e) => {
              e.preventDefault();
              void action(async () => {
                localStorage.setItem(
                  "pdeflow.training.pending-token",
                  token.current,
                );
                const started = await api<Run>("/training/start", {
                  ...request,
                  submission_token: token.current,
                });
                setRuns((old) => [
                  started,
                  ...old.filter((r) => r.id !== started.id),
                ]);
                chooseRun(started.id);
                token.current = crypto.randomUUID();
                localStorage.removeItem("pdeflow.training.pending-token");
              });
            }}
          >
            <div className="training-form-grid">
              <label>
                Training equations
                <select
                  aria-label="Training equations"
                  value={specialist}
                  onChange={(e) => {
                    setSpecialist(e.target.value);
                    setAdvanced((previous) => {
                      const next = { ...previous };
                      delete next.family_weights;
                      return next;
                    });
                  }}
                >
                  <option value="heat-2d">Heat only</option>
                  <option value="wave-2d">Wave only</option>
                  <option value="heat-wave-2d">Heat + Wave</option>
                </select>
              </label>
              {!focused && (
                <>
                  <label>
                    Model capacity
                    <select
                      aria-label="Model capacity"
                      defaultValue="budget"
                      onChange={(e) => {
                        const presets: Record<string, number[]> = {
                          smoke: [24, 1, 4, 8, 1, 1, 4],
                          xsmall: [256, 6, 8, 64, 2, 2, 16],
                          small: [384, 8, 8, 128, 3, 2, 24],
                          medium: [512, 12, 8, 256, 4, 3, 32],
                          large: [768, 16, 12, 384, 4, 3, 48],
                          xlarge: [1024, 24, 16, 512, 6, 4, 64],
                        };
                        const p = presets[e.target.value];
                        if (p)
                          setAdvanced((a) => ({
                            ...a,
                            width: p[0],
                            layers: p[1],
                            heads: p[2],
                            tokens: p[3],
                            joint: {
                              encoder_depth: p[4],
                              decoder_depth: p[5],
                              observation_tokens: p[6],
                            },
                          }));
                      }}
                    >
                      <option value="budget">Budget default</option>
                      {[
                        "smoke",
                        "xsmall",
                        "small",
                        "medium",
                        "large",
                        "xlarge",
                      ].map((x) => (
                        <option key={x} value={x}>
                          {x}
                        </option>
                      ))}
                      <option value="custom">Custom — edit Advanced</option>
                    </select>
                  </label>
                  <label>
                    Model
                    <select
                      aria-label="Training model"
                      value={model}
                      onChange={(e) => {
                        setModel(e.target.value);
                        resetAdvanced();
                      }}
                    >
                      <option value="resolution-flexible" disabled={false}>
                        Shared operator-aware latent model
                      </option>
                    </select>
                  </label>
                </>
              )}
              <label>
                Training budget
                <select
                  aria-label="Training budget"
                  value={budget}
                  onChange={(e) => {
                    setBudget(e.target.value);
                    resetAdvanced();
                  }}
                >
                  {Object.entries(budgetNames).map(([key, label]) => (
                    <option key={key} value={key}>
                      {label}
                    </option>
                  ))}
                </select>
              </label>
              <label>
                Device
                <select
                  aria-label="Training device"
                  value={device}
                  onChange={(e) => setDevice(e.target.value)}
                >
                  {Object.entries(deviceNames).map(([key, label]) => (
                    <option
                      key={key}
                      value={key}
                      disabled={key !== "auto" && !options?.devices[key]}
                    >
                      {label}
                      {key !== "auto" && !options?.devices[key]
                        ? " · unavailable"
                        : ""}
                    </option>
                  ))}
                </select>
              </label>
            </div>
            <p>
              Training equations are independent of the equation displayed in
              the playground.
            </p>
            <p>{options?.budgets[budget]}</p>
            {plan?.architecture_parameters && (
              <details>
                <summary>Model size and components</summary>
                <p>
                  {Object.entries(plan.architecture_parameters)
                    .filter(([, v]) => typeof v === "number")
                    .map(
                      ([k, v]) =>
                        `${k.replaceAll("_", " ")}: ${v.toLocaleString()}`,
                    )
                    .join(" · ")}
                  . Exact parameter counts; memory fit and accuracy require
                  measurement.
                </p>
              </details>
            )}
            <p>
              {options?.auto_explanation}{" "}
              {plan && (
                <>
                  Resolved device:{" "}
                  <strong>{deviceNames[plan.resolved_device]}</strong>.
                </>
              )}
            </p>
            {plan && (
              <>
                <div
                  className="training-contract"
                  data-testid="resolved-training-summary"
                  aria-busy={planPending}
                >
                  {planPending ? (
                    <p>Resolving training equations…</p>
                  ) : !planValid ? (
                    <p>
                      Training configuration is invalid. Resolve the error
                      before starting.
                    </p>
                  ) : (
                    <>
                      <strong>Resolved training equations</strong>
                      {plan.training_physics ? (
                        <>
                          {Object.entries(plan.training_physics.equations).map(
                            ([family, equation]) => (
                              <p key={family}>
                                <strong>
                                  {family === "heat" ? "Heat" : "Wave"}:
                                </strong>{" "}
                                {equation}
                              </p>
                            ),
                          )}
                          <p>
                            <strong>PDE family sampling weights:</strong>{" "}
                            {Object.entries(
                              plan.training_physics.family_weights,
                            )
                              .map(
                                ([f, w]) =>
                                  `${f === "heat" ? "Heat" : "Wave"} ${(100 * w).toLocaleString(undefined, { maximumFractionDigits: 2 })}%`,
                              )
                              .join(" · ")}
                          </p>
                          {Object.entries(
                            plan.training_physics.state_schemas,
                          ).map(([f]) => (
                            <p key={f}>
                              {f === "heat"
                                ? "Heat state: u."
                                : "Wave state: displacement u and velocity v = ∂u/∂t."}
                            </p>
                          ))}
                        </>
                      ) : (
                        <p>{plan.equation}</p>
                      )}
                      <p>
                        {plan.grid.shape.length} spatial dimensions · periodic
                        Cartesian domain ·{" "}
                        {plan.grid.domain.origins?.join(", ") || "origin 0"} +
                        extents {plan.grid.domain.lengths.join(" × ")} ·{" "}
                        {plan.grid.temporal_kind === "static"
                          ? "static; no physical time"
                          : `physical time τ ${plan.grid.times[0]}–${plan.grid.times.at(-1)}`}
                      </p>
                      <p>
                        <strong>Training distribution:</strong> {plan.varies}{" "}
                        This is a population of examples, separate from the
                        reference currently displayed in the workbench.
                      </p>
                      <p>
                        Active physical parameter ranges:{" "}
                        {parameterRanges(
                          plan.training_physics?.parameter_ranges ||
                            plan.distribution,
                          plan.problem,
                        )}
                        .
                      </p>
                    </>
                  )}
                </div>
                <details>
                  <summary>Budget details and resolved configuration</summary>
                  <p>
                    Reference grid {plan.grid.shape.join(" × ")} ·{" "}
                    {plan.grid.temporal_kind === "static"
                      ? "static"
                      : plan.config.temporal_mode === "continuous"
                        ? "Continuous physical-time training; display frames separate"
                        : `${plan.grid.times.length} saved physical times`}
                    . Training updates are separate from physical time and
                    generation integration steps.
                  </p>
                  <dl>
                    {Object.entries(plan.examples).map(([key, value]) => (
                      <div key={key}>
                        <dt>{key.replaceAll("_", " ")}</dt>
                        <dd>{value}</dd>
                      </div>
                    ))}
                    {Object.entries(plan.updates).map(([key, value]) => (
                      <div key={key}>
                        <dt>{stageNames[key]} updates</dt>
                        <dd>{value}</dd>
                      </div>
                    ))}
                  </dl>
                  <pre>{JSON.stringify(plan, null, 2)}</pre>
                </details>
                {model === "resolution-flexible" &&
                  ["heat-2d", "wave-2d", "heat-wave-2d"].includes(
                    specialist,
                  ) && (
                    <FieldLossControls
                      value={
                        (advanced.field_loss ||
                          plan.config.field_loss ||
                          {}) as Record<string, unknown>
                      }
                      onChange={(value) => setSetting("field_loss", value)}
                      request={request}
                      queryPoints={Number(current("query_points") || 256)}
                    />
                  )}
                <details>
                  <summary>Advanced settings and architecture</summary>
                  <p>
                    {model === "resolution-flexible"
                      ? "A1 state representation → A2 graph-conditioned learned evolution → B coupled initial-state/scalar generation → spatial decoder. Query flexibility does not establish fine-scale accuracy."
                      : "Field-space conditional flow on a fixed grid. Measurements guide inference rather than directly condition training."}
                  </p>
                  <div className="training-form-grid">
                    {model === "resolution-flexible" ? (
                      <>
                        {number("Training examples", "train_count")}
                        {number("Validation examples", "validation_count")}
                        {number("Untouched test examples", "test_count")}
                        {number("Dataset and training seed", "seed", 1, 0)}
                        {number("Representation updates", "ae_steps")}
                        {number("Evolution updates", "evolution_steps")}
                        {number("Evolution depth", "evolution.depth")}
                        {number(
                          "Latent transition weight",
                          "evolution.latent_weight",
                          0.01,
                          0,
                        )}
                        {number(
                          "Composition weight",
                          "evolution.composition_weight",
                          0.001,
                          0,
                        )}
                        {number("Flow training updates", "flow_steps")}
                        {number("Representation width", "width")}
                        {number("Latent tokens", "tokens")}
                        {number("Attention layers", "layers")}
                        {number("Attention heads", "heads")}
                        {number("Encoder depth", "joint.encoder_depth")}
                        {number("Decoder depth", "joint.decoder_depth")}
                        {number(
                          "Observation aggregation tokens",
                          "joint.observation_tokens",
                        )}
                        {number("Observation depth", "joint.observation_depth")}
                        {number("Context depth", "joint.context_depth")}
                        {number("Decoder query chunk", "joint.decoder_chunk")}
                        {number(
                          "Known scalar probability",
                          "joint.known_probability",
                          0.01,
                          0,
                        )}
                        {number("Encoder points", "encoder_points")}
                        {number("Training query points", "query_points")}
                        {number(
                          "Adam learning rate",
                          "learning_rate",
                          0.0001,
                          0.000001,
                        )}
                        {number(
                          "Validate and checkpoint every updates",
                          "checkpoint_every",
                        )}
                        {numbers("Reference grid shape", "reference_shape")}
                        {plan.config.temporal_mode === "continuous" ? (
                          <>
                            {numbers("Physical time interval", "time_interval")}
                            {number(
                              "Animation / export frames",
                              "animation_frames",
                            )}
                            <p>
                              Encoder count is spatial points per complete
                              state; query count is spatial supervision per
                              quantity/time. Initial/early/uniform/final
                              sampling is configured in Advanced JSON.
                            </p>
                          </>
                        ) : (
                          numbers("Saved physical times", "times")
                        )}
                        {specialist !== "wave-2d" &&
                          numbers("Diffusivity range", "diffusivity_range")}
                        {specialist !== "heat-2d" &&
                          numbers("Wave speed range", "wave_speed_range")}
                        {numbers(
                          "Observation count range",
                          "observation_counts",
                        )}
                        {numbers("Absolute noise sigma range", "noise_range")}
                        {number(
                          "Zero-observation probability",
                          "unconditional_probability",
                          0.01,
                          0.001,
                        )}
                        <label>
                          Maximum representation MSE / constant baseline
                          <input
                            aria-label="Representation gate ratio"
                            type="number"
                            min={0.01}
                            max={1}
                            step={0.01}
                            value={gateRatio}
                            onChange={(e) =>
                              setGateRatio(e.target.valueAsNumber)
                            }
                          />
                        </label>
                      </>
                    ) : (
                      <>
                        {number(
                          "Fixed-grid training updates",
                          "training.steps",
                        )}
                        {number("Effective batch", "training.batch_size")}
                        {number("Training seed", "training.seed", 1, 0)}
                        {number("Dataset seed", "data.seed", 1, 0)}
                        {number(
                          "Adam learning rate",
                          "training.learning_rate",
                          0.0001,
                          0.000001,
                        )}
                        {number(
                          "Validate and checkpoint every updates",
                          "training.checkpoint_every",
                        )}
                        {number("Model width", "model.width")}
                        {number("Model blocks", "model.blocks")}
                        <label>
                          Backbone
                          <select
                            aria-label="Fixed-grid backbone"
                            value={String(
                              current("model.name") || "residual_cnn",
                            )}
                            onChange={(e) =>
                              setAdvanced((old) => ({
                                ...old,
                                model: {
                                  ...((old.model as object) || {}),
                                  name: e.target.value,
                                  levels: 1,
                                },
                              }))
                            }
                          >
                            <option value="residual_cnn">Residual CNN</option>
                            <option value="multiscale_unet">
                              Multiscale U-Net
                            </option>
                          </select>
                        </label>
                        {current("model.name") === "multiscale_unet" &&
                          number("U-Net levels", "model.levels")}
                        {number(
                          "Calibration examples",
                          "provider.calibration_count",
                        )}
                        {number(
                          "Validation examples",
                          "provider.validation_count",
                        )}
                        {number(
                          "Untouched test examples",
                          "provider.test_count",
                        )}
                        <label>
                          Training example source
                          <select
                            aria-label="Training example source"
                            value={String(current("provider.mode") || "online")}
                            onChange={(e) =>
                              setSetting("provider.mode", e.target.value)
                            }
                          >
                            <option value="online">Generate online</option>
                            <option value="offline">
                              Fixed offline dataset
                            </option>
                            <option value="hybrid">
                              Bounded reusable pool
                            </option>
                          </select>
                        </label>
                        {current("provider.mode") === "offline" &&
                          number(
                            "Offline training examples",
                            "provider.offline_count",
                          )}
                        {current("provider.mode") === "hybrid" && (
                          <>
                            {number(
                              "Reusable example pool",
                              "provider.capacity",
                            )}
                            {number(
                              "Example reuse limit",
                              "provider.reuse_limit",
                            )}
                          </>
                        )}
                        {numbers(
                          "Reference and model grid shape",
                          "grid.shape",
                        )}
                        {plan.grid.temporal_kind !== "static" &&
                          numbers("Saved physical times", "grid.times")}
                        {number(
                          "Validation sensor locations",
                          "inference.sensors",
                        )}
                        {number(
                          "Validation absolute noise sigma",
                          "inference.noise_std",
                          0.01,
                          0,
                        )}
                        {number(
                          "Validation generation steps",
                          "inference.steps",
                        )}
                      </>
                    )}
                  </div>
                  {model === "resolution-flexible" && (
                    <>
                      <p>
                        The default quality gate requires validation MSE below a
                        training-fitted constant predictor. This is a minimal
                        sanity check, not a universal quality threshold.
                        Pipeline checks exercise all three stages regardless and
                        retain the warning.
                      </p>
                      <label className="checkbox-label">
                        <input
                          type="checkbox"
                          checked={allowPoor}
                          onChange={(e) => setAllowPoor(e.target.checked)}
                        />
                        Explicitly continue A2/B despite a poor representation;
                        record my override
                      </label>
                    </>
                  )}
                </details>
              </>
            )}
            {plan?.batch_settings && (
              <BatchControls
                request={request}
                batches={plan.batch_settings}
                onApply={setAdvanced}
                onDirty={setBatchDirty}
                onRunning={setBatchBenchmarkRunning}
                active={active || busy}
              />
            )}
            <Button
              type="submit"
              disabled={
                busy ||
                batchDirty ||
                batchBenchmarkRunning ||
                !runsLoaded ||
                planPending ||
                !planValid ||
                !plan ||
                active
              }
            >
              Train model
            </Button>
            {active && (
              <p>
                A training run is active. Inspect or safely stop it below before
                submitting another.
              </p>
            )}
            <p>
              Opening this screen never trains. Lower training loss alone does
              not prove reconstruction quality.
            </p>
          </form>
        </CardContent>
      </Card>
      {error && <p role="alert">{error}</p>}
      <Card>
        <CardHeader>
          <CardTitle>Training progress and results</CardTitle>
        </CardHeader>
        <CardContent>
          {!runsLoaded ? (
            <p role="status">Loading saved training runs…</p>
          ) : !runs.length ? (
            <p>No integrated training run yet.</p>
          ) : (
            <>
              <label>
                Run
                <select
                  aria-label="Integrated training run"
                  value={selected}
                  onChange={(e) => chooseRun(e.target.value)}
                >
                  {runs.map((r) => (
                    <option key={r.id} value={r.id}>
                      {r.plan.problem} · {r.plan.model_label} · {r.state} ·{" "}
                      {r.id}
                    </option>
                  ))}
                </select>
              </label>
              {run && (
                <>
                  <p role="status">
                    <strong>
                      {run.state} · {run.stage_label}
                    </strong>{" "}
                    ·{" "}
                    {run.stage_budget > 0
                      ? `${run.progress}/${run.stage_budget} updates in current learning stage`
                      : "No optimizer running in this stage"}{" "}
                    · elapsed {run.elapsed_seconds.toFixed(1)} s
                  </p>
                  <p>
                    {Object.entries(run.plan.updates)
                      .map(
                        ([stage, total]) =>
                          `${stageNames[stage]}: ${run.finished_stages.includes(stage) ? total : stage === run.stage ? run.progress : 0}/${total} updates`,
                      )
                      .join(" · ")}
                  </p>
                  {run.training_examples_consumed != null && (
                    <p>
                      {run.training_examples_consumed} unique physical training
                      examples consumed; calibration and validation are
                      separate.
                    </p>
                  )}
                  <p>{run.stage_meaning}</p>
                  <p>{run.meaning}</p>
                  <ol className="training-stages">
                    {run.plan.stages.map((s) => (
                      <li
                        key={s}
                        aria-current={s === run.stage ? "step" : undefined}
                      >
                        {run.finished_stages.includes(s) ? "✓ " : ""}
                        {stageNames[s]}
                      </li>
                    ))}
                  </ol>
                  <p>
                    {run.eta_seconds === null
                      ? "ETA unavailable until enough timed updates have been measured."
                      : `Estimated remaining time in this learning stage: ${Math.ceil(run.eta_seconds)} s.`}
                  </p>
                  {run.history.length > 0 && (
                    <LineChart
                      title="Recent learning objectives"
                      xlabel="Optimizer update"
                      ylabel="Stage objective; not reconstruction error"
                      series={[
                        {
                          name: "Training",
                          color: "#426eab",
                          points: run.history.map((h) => ({
                            x: h.step,
                            y: h.training_loss,
                          })),
                        },
                        {
                          name: "Fixed validation objective",
                          color: "#aa6b38",
                          dashed: true,
                          points: run.history
                            .filter((h) => h.validation_loss !== undefined)
                            .map((h) => ({ x: h.step, y: h.validation_loss! })),
                        },
                      ]}
                    />
                  )}
                  <p>
                    Training loss measures the current optimization objective.
                    Validation loss uses held-out examples with fixed
                    randomness. Physical reconstruction errors below measure a
                    different task.
                  </p>
                  {run.error && <p role="alert">{run.error}</p>}
                  {run.warning && <p>{run.warning}</p>}
                  {run.representation_quality && (
                    <p>
                      Representation validation MSE:{" "}
                      {run.representation_quality.validation_mse.toPrecision(4)}{" "}
                      · constant baseline:{" "}
                      {run.representation_quality.constant_baseline_mse.toPrecision(
                        4,
                      )}{" "}
                      · ratio{" "}
                      {run.representation_quality.mse_ratio?.toPrecision(4) ??
                        "unavailable"}
                      .{" "}
                      {run.representation_quality.passed
                        ? "Minimal baseline check passed; inspect remaining distortion."
                        : "Poor representation: minimal baseline check failed."}
                    </p>
                  )}
                  {validationSummary(run.validation).map((line) => (
                    <p key={line}>{line}</p>
                  ))}
                  {["queued", "running", "stopping"].includes(run.state) && (
                    <Button
                      variant="outline"
                      disabled={busy || run.stop_requested}
                      onClick={() =>
                        void action(async () => {
                          await api(`/training/runs/${run.id}/stop`, {});
                        })
                      }
                    >
                      {run.stop_requested
                        ? "Safe stop requested"
                        : "Stop safely"}
                    </Button>
                  )}
                  {run.stop_requested && (
                    <p>
                      Preparation or evaluation may finish first. Learning stops
                      at a safe update boundary and saves resumable state.
                    </p>
                  )}
                  {run.can_resume && (
                    <Button
                      disabled={busy}
                      onClick={() =>
                        void action(async () => {
                          await api(`/training/runs/${run.id}/resume`, {
                            allow_poor_representation: continuePoor,
                          });
                        })
                      }
                    >
                      Resume training
                      {continuePoor ? " with recorded quality override" : ""}
                    </Button>
                  )}
                  {run.state === "needs_review" && (
                    <>
                      <p>
                        Inspect representation errors before continuing. These
                        errors use complete reference context and are separate
                        from observation-only reconstruction.
                      </p>
                      <label className="checkbox-label">
                        <input
                          type="checkbox"
                          checked={continuePoor}
                          onChange={(e) => setContinuePoor(e.target.checked)}
                        />
                        Continue this run despite representation distortion;
                        record my override
                      </label>
                    </>
                  )}
                  <details>
                    <summary>Logs, stage metrics and technical names</summary>
                    <pre>
                      {run.log_tail}
                      {"\n"}
                      {run.stage_log_tail}
                    </pre>
                    <pre>
                      {JSON.stringify(
                        {
                          stage: run.stage,
                          representation: run.representation_validation,
                          validation: run.validation,
                          eta_scope: run.eta_scope,
                        },
                        null,
                        2,
                      )}
                    </pre>
                  </details>
                  {run.state === "completed" && (
                    <>
                      <p>
                        Configured stages completed. The inference artifact is
                        available; it has not been automatically selected for
                        reconstruction.
                      </p>
                      <p>
                        Validation previews use a fixed held-out case. Test data
                        has not been evaluated.
                      </p>
                      <div className="training-form-grid">
                        <label>
                          Validation physical-time frame
                          <input
                            aria-label="Validation physical-time frame"
                            type="number"
                            min={0}
                            max={run.plan.grid.times.length - 1}
                            value={physicalTime}
                            onChange={(e) =>
                              setPhysicalTime(e.target.valueAsNumber)
                            }
                            disabled={run.plan.grid.temporal_kind === "static"}
                          />
                        </label>
                        <label>
                          Validation field index
                          <input
                            aria-label="Validation field index"
                            type="number"
                            min={0}
                            max={
                              run.plan.problem.startsWith("chemistry")
                                ? 2
                                : run.plan.problem.startsWith("darcy")
                                  ? 1
                                  : 0
                            }
                            value={field}
                            onChange={(e) => setField(e.target.valueAsNumber)}
                          />
                        </label>
                        {run.plan.grid.shape.length === 3 && (
                          <>
                            <label>
                              Validation slice axis
                              <select
                                aria-label="Validation slice axis"
                                value={axis}
                                onChange={(e) => {
                                  setAxis(Number(e.target.value));
                                  setIndex(0);
                                }}
                              >
                                <option value={0}>x</option>
                                <option value={1}>y</option>
                                <option value={2}>z</option>
                              </select>
                            </label>
                            <label>
                              Validation slice index
                              <input
                                aria-label="Validation slice index"
                                type="number"
                                min={0}
                                max={run.plan.grid.shape[axis] - 1}
                                value={index}
                                onChange={(e) =>
                                  setIndex(e.target.valueAsNumber)
                                }
                              />
                            </label>
                          </>
                        )}
                      </div>
                      {slice && (
                        <div className="training-validation-plots">
                          <FieldPlot
                            slice={slice}
                            values={slice.reference}
                            title="Reference"
                          />
                          <FieldPlot
                            slice={slice}
                            values={slice.values}
                            title="Reconstruction"
                          />
                          <FieldPlot
                            slice={slice}
                            values={slice.error}
                            title="Absolute error"
                          />
                        </div>
                      )}
                      {ownModel && (
                        <Button
                          onClick={() => void action(() => use(ownModel))}
                          disabled={busy}
                        >
                          Use this model
                        </Button>
                      )}
                      {ownModel && !ownModel.compatibility.compatible && (
                        <p>
                          Using this model loads its training problem and clears
                          incompatible measurements. The previous mathematical
                          draft remains available through Undo.
                        </p>
                      )}
                    </>
                  )}
                </>
              )}
            </>
          )}
        </CardContent>
      </Card>
      <Card>
        <CardHeader>
          <CardTitle>Your models</CardTitle>
        </CardHeader>
        <CardContent>
          <p>
            Both model types and previously saved exports appear here. Models in
            project output workspaces are discovered without moving their files.
          </p>
          <Button
            variant="outline"
            onClick={() => {
              setRevision((v) => v + 1);
              onRefresh();
            }}
          >
            Refresh model list
          </Button>
          {!modelsLoaded ? (
            <p>Discovering saved models…</p>
          ) : (
            !models.length && <p>No usable inference artifacts yet.</p>
          )}
          <div className="model-list">
            {models.map((m) => (
              <article className="training-model-row" key={m.id}>
                <h3>
                  {m.problem.family} · {m.problem.grid.shape.length}D · {m.type}
                </h3>
                <p>
                  {m.status} ·{" "}
                  {Object.entries(m.updates)
                    .map(
                      ([stage, value]) =>
                        `${stageNames[stage]} ${value} updates`,
                    )
                    .join(" · ")}
                </p>
                <p>
                  <strong>
                    {m.pipeline_check
                      ? "Pipeline-check checkpoint"
                      : "Training experiment"}
                  </strong>{" "}
                  · {m.quality}
                </p>
                {validationSummary(m.validation).map((line) => (
                  <p key={line}>{line}</p>
                ))}
                <p>
                  {m.compatibility.compatible
                    ? "Compatible with current problem."
                    : "Current problem differs."}{" "}
                  {m.compatibility.summary}
                </p>
                <details>
                  <summary>Validation results and model identity</summary>
                  <p>
                    {m.validation
                      ? "Held-out validation; no test-based selection."
                      : "No validation reconstruction report is attached; quality is unknown."}
                  </p>
                  <pre>
                    {JSON.stringify(
                      {
                        id: m.id,
                        updates: m.updates,
                        validation: m.validation,
                      },
                      null,
                      2,
                    )}
                  </pre>
                </details>
                <Button
                  disabled={busy}
                  onClick={() => void action(() => use(m))}
                >
                  {m.compatibility.compatible
                    ? "Use for reconstruction"
                    : "Use with training problem"}
                </Button>
                {!m.compatibility.compatible && (
                  <small>
                    Loads this model's physical problem; incompatible
                    measurements are cleared.
                  </small>
                )}
                {m.workflow_id && (
                  <Button
                    variant="ghost"
                    onClick={() => chooseRun(m.workflow_id!)}
                  >
                    Inspect training run
                  </Button>
                )}
                {m.can_extend && m.workflow_id && (
                  <details>
                    <summary>Train further · fixed-grid continuation</summary>
                    <p>
                      Add optimizer updates with the same training distribution.
                      The previous completed model and validation artifacts are
                      preserved. Repeated validation is for development; test
                      data stays separate.
                    </p>
                    <label>
                      Additional optimizer updates
                      <input
                        aria-label={`Additional optimizer updates ${m.id}`}
                        type="number"
                        min={1}
                        max={1000000}
                        value={additionalUpdates}
                        onChange={(e) =>
                          setAdditionalUpdates(e.target.valueAsNumber)
                        }
                      />
                    </label>
                    <Button
                      disabled={busy || active}
                      onClick={() =>
                        void action(async () => {
                          const next = await api<Run>(
                            `/training/runs/${m.workflow_id}/extend`,
                            { additional_updates: additionalUpdates },
                          );
                          setRuns((old) => [
                            next,
                            ...old.filter((r) => r.id !== next.id),
                          ]);
                          chooseRun(next.id);
                        })
                      }
                    >
                      Train further
                    </Button>
                  </details>
                )}
              </article>
            ))}
          </div>
        </CardContent>
      </Card>
    </section>
  );
}
