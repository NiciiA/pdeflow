import { useEffect, useMemo, useState } from "react";
import { api } from "./types";
import type { Problem } from "./types";
import type { TrainingSelection } from "./TrainModel";
import { FieldLossControls } from "./FieldLossControls";
import { HeatCanvas } from "./HeatCanvas";
import { LineChart } from "./ScienceCharts";

const meshModels: Record<string, string> = {
  mesh_m1_v1: "M1 · Deterministic initial-state mesh — recommended first",
  mesh_m2_v1: "M2 · Deterministic mesh + evolution",
  mesh_m3_fgn_v1: "M3 · FGN-style mesh — first generative candidate",
  mesh_m4_flow_v1: "M4 · Conditional physical-state flow",
  mesh_m5_diffusion_v1: "M5 · Conditional physical-state diffusion",
};
type StageSettings = {
  learning_rate: number;
  effective_batch: number;
  microbatch: number;
  accumulation: number;
  last_microbatch: number;
  optimizer: { schedule?: string; warmup?: number };
  schedule_horizon: number;
};
type StageOverride = {
  optimizer?: { schedule?: string; warmup?: number };
  learning_rate?: number;
  batch_size?: number;
  microbatch?: number;
};
type BatchReport = {
  candidates: {
    microbatch: number;
    state: string;
    error?: string;
    report?: {
      estimated_total_training_seconds: number;
      stages: Record<
        string,
        {
          examples_per_second: number;
          end_to_end_seconds_per_successful_update: number;
          successful_updates: number;
          skipped_updates: number;
          memory: { cuda_peak_allocated: number | null };
        }
      >;
    };
  }[];
};
type Plan = {
  training_settings?: { policy: string; stages: Record<string, StageSettings> };
  model_label: string;
  scope: string;
  updates: Record<string, number>;
  stages: string[];
  config: {
    field_loss: Record<string, unknown>;
    data: { query_points?: number };
    node_shape: number[];
    width: number;
    depth: number;
    sampler: Record<string, unknown>;
  };
  architecture_parameters: Record<string, number | string>;
  training_physics: {
    equations: Record<string, string>;
    family_weights: Record<string, number>;
    parameter_ranges?: Record<string, number[]>;
    state_schemas: Record<string, string[]>;
  };
  resolved_device: string;
  pipeline_check: boolean;
};
type Preview = {
  values: number[][];
  reference: number[][];
  error: number[][];
  vmin: number;
  vmax: number;
  checkpoint_update: number;
  preview_label: {
    family: string;
    quantity: string;
    physical_time: number;
    scope: string;
    rmse: number;
  };
};
type Run = {
  id: string;
  state: string;
  stage: string;
  progress: number;
  stage_budget: number;
  error?: string;
  warning?: string;
  model_id?: string;
  can_resume: boolean;
  plan: { model: string; model_label: string };
  history: {
    step: number;
    training_loss: number;
    validation_loss?: number;
    components?: Record<string, number>;
  }[];
};

export function MeshTraining({
  problem,
  onUse,
  onRefresh,
  onLegacy,
}: {
  problem?: Problem;
  onUse?: (s: TrainingSelection) => void;
  onRefresh: () => void;
  onLegacy: () => void;
}) {
  const [model, setModel] = useState("mesh_m1_v1"),
    [capacity, setCapacity] = useState("xsmall"),
    [budget, setBudget] = useState("development"),
    [family, setFamily] = useState("heat-wave-2d"),
    [device, setDevice] = useState("auto"),
    [learningRate, setLearningRate] = useState(0.001),
    [batchPolicy, setBatchPolicy] = useState("keep_updates"),
    [stageSettings, setStageSettings] = useState<Record<string, StageOverride>>(
      {},
    ),
    [benchmarkCandidates, setBenchmarkCandidates] = useState("1,2,4,8"),
    [batchReport, setBatchReport] = useState<{
      key: string;
      value: BatchReport;
    } | null>(null),
    [batch, setBatch] = useState(8),
    [micro, setMicro] = useState(8),
    [loss, setLoss] = useState<Record<string, unknown>>({ type: "mse" }),
    [advanced, setAdvanced] = useState("{}"),
    [draft, setDraft] = useState("{}"),
    [previewRecord, setPreview] = useState<{
      key: string;
      value: Preview;
    } | null>(null),
    [planRecord, setPlan] = useState<{ key: string; value: Plan } | null>(null),
    [runs, setRuns] = useState<Run[]>([]),
    [selected, setSelected] = useState(""),
    [error, setError] = useState(""),
    [busy, setBusy] = useState(false),
    [profileRecord, setProfile] = useState<{
      key: string;
      value: unknown;
    } | null>(null);
  const request = useMemo(
    () => ({
      problem: family,
      model,
      budget,
      device,
      advanced: {
        ...JSON.parse(advanced || "{}"),
        capacity,
        learning_rate: learningRate,
        batch_policy: batchPolicy,
        stage_settings: stageSettings,
        batch_size: batch,
        microbatch: micro,
        field_loss: loss,
      },
    }),
    [
      family,
      model,
      budget,
      device,
      capacity,
      batch,
      micro,
      learningRate,
      batchPolicy,
      stageSettings,
      loss,
      advanced,
    ],
  );
  const run = runs.find((r) => r.id === selected);
  const requestKey = JSON.stringify(request),
    previewKey = `${selected}:${run?.stage}:${run?.progress}:${run?.state}`;
  const plan = planRecord?.key === requestKey ? planRecord.value : null,
    profile = profileRecord?.key === requestKey ? profileRecord.value : null,
    preview = previewRecord?.key === previewKey ? previewRecord.value : null;
  useEffect(() => {
    let live = true;
    api<Plan>("/training/plan", request)
      .then((p) => {
        if (live) {
          setPlan({ key: requestKey, value: p });
          setError("");
        }
      })
      .catch((e) => {
        if (live) setError(String(e.message));
      });
    return () => {
      live = false;
    };
  }, [request, requestKey]);
  useEffect(() => {
    let live = true;
    async function refresh() {
      try {
        const result = await api<Run[]>("/training/runs");
        if (live) setRuns(result);
      } catch (e) {
        if (live) setError(String((e as Error).message));
      }
    }
    void refresh();
    const timer = setInterval(() => void refresh(), 1500);
    return () => {
      live = false;
      clearInterval(timer);
    };
  }, []);
  useEffect(() => {
    let live = true;
    if (selected)
      api<Preview>(`/training/runs/${selected}/preview`)
        .then((p) => {
          if (live) setPreview({ key: previewKey, value: p });
        })
        .catch(() => {});
    return () => {
      live = false;
    };
  }, [selected, previewKey]);
  async function act(callback: () => Promise<void>) {
    setError("");
    setBusy(true);
    try {
      await callback();
      setRuns(await api<Run[]>("/training/runs"));
      onRefresh();
    } catch (e) {
      setError(String((e as Error).message));
    } finally {
      setBusy(false);
    }
  }
  return (
    <section className="integrated-training">
      <h1>Train a mesh model</h1>
      <p>
        Learn physical fields directly from sparse, time-stamped measurements.
        No separately trained full-field autoencoder is required.
      </p>
      <div className="training-form-grid">
        <label>
          Model
          <select
            aria-label="New model candidate"
            value={model}
            onChange={(e) => setModel(e.target.value)}
          >
            {Object.entries(meshModels).map(([id, name]) => (
              <option key={id} value={id}>
                {name}
              </option>
            ))}
          </select>
        </label>
        <label>
          Training equations
          <select
            aria-label="Training equations"
            value={family}
            onChange={(e) => setFamily(e.target.value)}
          >
            <option value="heat-wave-2d">Heat + Wave</option>
            <option value="heat-2d">Heat only</option>
            <option value="wave-2d">Wave only</option>
          </select>
        </label>
        <label>
          Model capacity
          <select
            aria-label="Model capacity"
            value={capacity}
            onChange={(e) => setCapacity(e.target.value)}
          >
            {[
              "pipeline-check",
              "xsmall",
              "small",
              "medium",
              "large",
              "xlarge",
            ].map((x) => (
              <option key={x}>{x}</option>
            ))}
          </select>
        </label>
        <label>
          Training budget
          <select
            aria-label="Training budget"
            value={budget}
            onChange={(e) => setBudget(e.target.value)}
          >
            <option value="pipeline-check">
              Pipeline check · execution only
            </option>
            <option value="development">Development run</option>
            <option value="extended">Extended run</option>
          </select>
        </label>
        <label>
          Device
          <select
            aria-label="Training device"
            value={device}
            onChange={(e) => setDevice(e.target.value)}
          >
            <option value="auto">Auto</option>
            <option value="cpu">CPU</option>
            <option value="mps">Apple MPS</option>
            <option value="cuda">NVIDIA CUDA</option>
          </select>
        </label>
      </div>
      <p>
        Training equations are independent of the displayed playground. Capacity
        and successful-update budget are separate. No model is claimed accurate
        merely because training completes.
      </p>
      {plan && (
        <div className="training-contract" data-testid="mesh-resolved-summary">
          <strong>{plan.model_label}</strong>
          <p>{plan.scope}</p>
          {Object.entries(plan.training_physics.equations).map(([f, e]) => (
            <p key={f}>
              {f}: {e} · sampling weight{" "}
              {plan.training_physics.family_weights[f]} · state{" "}
              {plan.training_physics.state_schemas[f].join(", ")}
            </p>
          ))}
          {Object.entries(plan.training_physics.parameter_ranges || {}).map(
            ([name, range]) => (
              <p key={name}>
                {name} training range: {range.join(" to ")}
              </p>
            ),
          )}
          <p>
            Initial-field shape mixtures are not PDE families. Measurement times
            are known; physical query time and generative noise/flow time are
            separate.
          </p>
          <p>
            Stages: {plan.stages.join(" → ")}. Updates:{" "}
            {Object.entries(plan.updates)
              .map(([k, v]) => `${k}: ${v}`)
              .join(" · ")}
            . Resolved device: {plan.resolved_device}.
          </p>
          <p>
            {plan.architecture_parameters.total?.toLocaleString()} parameters ·
            internal nodes {plan.config.node_shape.join(" × ")} · output
            coordinates are independent of node count; finer queries do not
            establish more accurate detail.
          </p>
        </div>
      )}
      <FieldLossControls
        modelFamily="mesh"
        value={loss}
        onChange={setLoss}
        request={request}
        queryPoints={plan?.config.data.query_points || 64}
      />
      <p>
        Field losses train deterministic/zero-noise branches and dynamics. FGN
        uses fair CRPS; flow/diffusion use distinct physical-state objectives.
        Raw total losses across these candidates are not directly comparable.
      </p>
      <h3>Optimizer and batch</h3>
      <div className="number-grid">
        <label>
          Learning rate
          <input
            aria-label="Mesh learning rate"
            type="number"
            min={0.000000001}
            step="any"
            value={learningRate}
            onChange={(e) => setLearningRate(e.target.valueAsNumber)}
          />
        </label>
        <label>
          Effective batch
          <input
            aria-label="Mesh effective batch"
            type="number"
            min={1}
            value={batch}
            onChange={(e) => setBatch(e.target.valueAsNumber)}
          />
        </label>
        <label>
          GPU microbatch
          <input
            aria-label="Mesh forward microbatch"
            type="number"
            min={1}
            max={batch}
            value={micro}
            onChange={(e) => setMicro(e.target.valueAsNumber)}
          />
        </label>
        <label>
          Batch budget policy
          <select
            aria-label="Mesh batch budget policy"
            value={batchPolicy}
            onChange={(e) => setBatchPolicy(e.target.value)}
          >
            <option value="keep_updates">Keep successful updates</option>
            <option value="keep_examples">
              Keep example budget (round up)
            </option>
          </select>
        </label>
      </div>
      <p>
        Derived accumulation: {Math.ceil(batch / micro)} forward/backward passes
        per optimizer update; final pass{" "}
        {batch - (Math.ceil(batch / micro) - 1) * micro} examples. Per-example
        losses sum / effective batch. GPU microbatch also applies on CPU/MPS.
      </p>
      <p>
        Learning rate 0.0001 (1e-4) is an available experiment, not a proven
        stability fix. No automatic LR or warmup scaling. These controls
        configure a NEW run; Resume keeps its recorded settings.
      </p>
      {plan?.training_settings && (
        <div data-testid="mesh-optimizer-summary">
          <strong>
            Resolved AdamW · weight decay 0 · {plan.training_settings.policy}
          </strong>
          <table>
            <thead>
              <tr>
                <th>Stage</th>
                <th>Learning rate</th>
                <th>Effective batch</th>
                <th>Microbatch</th>
                <th>Accumulation</th>
                <th>Schedule / warmup / horizon</th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(plan.training_settings.stages).map(
                ([stage, r]) => (
                  <tr key={stage}>
                    <td>{stage}</td>
                    <td>{r.learning_rate}</td>
                    <td>{r.effective_batch}</td>
                    <td>{r.microbatch}</td>
                    <td>
                      {r.accumulation} passes; last {r.last_microbatch}
                    </td>
                    <td>
                      {r.optimizer.schedule || "constant"} /{" "}
                      {r.optimizer.warmup || 0} / {r.schedule_horizon}
                    </td>
                  </tr>
                ),
              )}
            </tbody>
          </table>
        </div>
      )}
      <details>
        <summary>Advanced model and per-stage settings</summary>
        <p>
          Optional per-stage overrides. Unchecked stages inherit the controls
          above. Optional stage schedules and warmup are explicit
          successful-update settings; use optimizer in the JSON below for a
          shared schedule.
        </p>
        {Object.keys(planRecord?.value.training_settings?.stages || {}).map(
          (stage) => (
            <div key={stage}>
              <label>
                <input
                  type="checkbox"
                  aria-label={`Override ${stage}`}
                  checked={!!stageSettings[stage]}
                  onChange={(e) =>
                    setStageSettings((prev) => {
                      const next = { ...prev };
                      if (e.target.checked) next[stage] = {};
                      else delete next[stage];
                      return next;
                    })
                  }
                />
                Override {stage}
              </label>
              {stageSettings[stage] && (
                <div className="number-grid">
                  {(
                    [
                      ["learning_rate", "Learning rate", learningRate],
                      ["batch_size", "Effective batch", batch],
                      ["microbatch", "GPU microbatch", micro],
                    ] as const
                  ).map(([key, label, fallback]) => (
                    <label key={key}>
                      {label}
                      <input
                        type="number"
                        min={key === "learning_rate" ? 1e-9 : 1}
                        step={key === "learning_rate" ? "any" : 1}
                        aria-label={`${stage} ${label}`}
                        value={stageSettings[stage][key] ?? fallback}
                        onChange={(e) =>
                          setStageSettings((prev) => ({
                            ...prev,
                            [stage]: {
                              ...prev[stage],
                              [key]: e.target.valueAsNumber,
                            },
                          }))
                        }
                      />
                    </label>
                  ))}
                  <label>
                    Schedule
                    <select
                      aria-label={`${stage} schedule`}
                      value={
                        stageSettings[stage].optimizer?.schedule ?? "inherit"
                      }
                      onChange={(e) =>
                        setStageSettings((prev) => {
                          const opt = { ...prev[stage].optimizer };
                          if (e.target.value === "inherit") delete opt.schedule;
                          else opt.schedule = e.target.value;
                          return {
                            ...prev,
                            [stage]: { ...prev[stage], optimizer: opt },
                          };
                        })
                      }
                    >
                      <option value="inherit">Inherit shared schedule</option>
                      <option value="constant">Constant</option>
                      <option value="cosine">Cosine</option>
                    </select>
                  </label>
                  <label>
                    Warmup updates
                    <input
                      aria-label={`${stage} warmup`}
                      type="number"
                      min={0}
                      placeholder="Inherit shared warmup"
                      value={stageSettings[stage].optimizer?.warmup ?? ""}
                      onChange={(e) =>
                        setStageSettings((prev) => {
                          const opt = { ...prev[stage].optimizer };
                          if (e.target.value === "") delete opt.warmup;
                          else opt.warmup = e.target.valueAsNumber;
                          return {
                            ...prev,
                            [stage]: { ...prev[stage], optimizer: opt },
                          };
                        })
                      }
                    />
                  </label>
                </div>
              )}
            </div>
          ),
        )}
        <p>
          Other model-specific noise, objective weights, optimizer schedule,
          corruption and samplers are optional JSON overrides. Explicit controls
          above take precedence.
        </p>
        <textarea
          aria-label="Mesh advanced settings"
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
        />
        <button
          onClick={() => {
            try {
              const value = JSON.parse(draft);
              if (!value || Array.isArray(value) || typeof value !== "object")
                throw new Error("Expected JSON object");
              for (const key of [
                "learning_rate",
                "batch_size",
                "microbatch",
                "stage_settings",
                "batch_policy",
                "batch_reference",
                "capacity",
                "field_loss",
              ])
                if (key in value)
                  throw new Error(
                    `Use the explicit control for ${key}; JSON cannot override it.`,
                  );
              setAdvanced(draft);
              setError("");
            } catch (e) {
              setError(String((e as Error).message));
            }
          }}
        >
          Apply advanced settings
        </button>
        <pre>{plan ? JSON.stringify(plan.config, null, 2) : "Resolving…"}</pre>
      </details>
      <details>
        <summary>Bounded GPU microbatch benchmark</summary>
        <p>
          Runs the selected architecture and objective, including both FGN
          ensemble members. Eight successful updates per stage/candidate: two
          warmup, six measured. Effective batch, learning rate, points, update
          budget and scheduler are retained. No training run is modified. Set an
          explicit effective batch of at least 32 to test 8,16,32; available
          host RAM does not determine fit.
        </p>
        <label>
          Microbatch candidates
          <input
            aria-label="Microbatch candidates"
            value={benchmarkCandidates}
            onChange={(e) => setBenchmarkCandidates(e.target.value)}
          />
        </label>
        <button
          disabled={!plan || busy}
          onClick={() =>
            void act(async () =>
              setBatchReport({
                key: requestKey,
                value: await api<BatchReport>("/mesh/microbatch-benchmark", {
                  request,
                  candidates: benchmarkCandidates
                    .split(",")
                    .map((x) => Number(x.trim())),
                  updates: 8,
                  warmup: 2,
                }),
              }),
            )
          }
        >
          Benchmark microbatches
        </button>
        {batchReport?.key === requestKey && (
          <>
            <table>
              <thead>
                <tr>
                  <th>Microbatch / stage</th>
                  <th>Examples/s (steady)</th>
                  <th>End-to-end s/update</th>
                  <th>GPU peak bytes</th>
                  <th>Successful / skipped</th>
                  <th>Estimated total</th>
                </tr>
              </thead>
              <tbody>
                {batchReport.value.candidates.flatMap((candidate) =>
                  candidate.report
                    ? Object.entries(candidate.report.stages).map(
                        ([stage, r]) => (
                          <tr key={`${candidate.microbatch}:${stage}`}>
                            <td>
                              {candidate.microbatch} / {stage}
                            </td>
                            <td>{r.examples_per_second.toFixed(2)}</td>
                            <td>
                              {r.end_to_end_seconds_per_successful_update.toFixed(
                                3,
                              )}
                            </td>
                            <td>
                              {r.memory.cuda_peak_allocated ??
                                "Unavailable on this device"}
                            </td>
                            <td>
                              {r.successful_updates} / {r.skipped_updates}
                            </td>
                            <td>
                              {(
                                candidate.report!
                                  .estimated_total_training_seconds / 3600
                              ).toFixed(2)}{" "}
                              hours (all stages)
                            </td>
                          </tr>
                        ),
                      )
                    : [
                        <tr key={candidate.microbatch}>
                          <td>{candidate.microbatch}</td>
                          <td colSpan={5}>
                            {candidate.state}: {candidate.error}
                          </td>
                        </tr>,
                      ],
                )}
              </tbody>
            </table>
            <p>
              Timing projections are short-sample estimates. Setup, cold/warm
              timings, variability and excluded export/Drive costs are in
              Details. Apply a measured microbatch manually above, then
              preflight the exact final configuration.
            </p>
            <details>
              <summary>Benchmark details</summary>
              <pre>{JSON.stringify(batchReport.value, null, 2)}</pre>
            </details>
          </>
        )}
      </details>
      <button
        disabled={!plan || busy}
        onClick={() =>
          void act(async () =>
            setProfile({
              key: requestKey,
              value: await api("/mesh/preflight", request),
            }),
          )
        }
      >
        Preflight selected model
      </button>
      {profile != null && (
        <details open>
          <summary>Actual candidate preflight</summary>
          <pre>{JSON.stringify(profile, null, 2)}</pre>
        </details>
      )}
      <button
        disabled={!plan || busy}
        onClick={() =>
          void act(async () => {
            const r = await api<Run>("/training/start", {
              ...request,
              submission_token: crypto.randomUUID(),
            });
            setSelected(r.id);
          })
        }
      >
        Train model
      </button>
      {error && <p role="alert">{error}</p>}
      <h2>Training progress and saved runs</h2>
      <label>
        Training run
        <select
          aria-label="Mesh training run"
          value={selected}
          onChange={(e) => setSelected(e.target.value)}
        >
          <option value="">Select a recorded run</option>
          {runs.map((r) => (
            <option key={r.id} value={r.id}>
              {r.plan.model_label} · {r.state} · {r.id}
            </option>
          ))}
        </select>
      </label>
      {run && (
        <>
          <p>
            {run.plan.model_label} · {run.state} · {run.stage} · {run.progress}/
            {run.stage_budget} successful updates
          </p>
          {run.error && <p role="alert">{run.error}</p>}
          {run.warning && <p>{run.warning}</p>}
          {run.history.length > 0 && (
            <>
              <LineChart
                title="Current stage objective — not comparable across candidate families"
                xlabel="Successful update"
                ylabel="Objective"
                series={[
                  {
                    name: "Training objective",
                    points: run.history.map((r) => ({
                      x: r.step,
                      y: r.training_loss,
                    })),
                    color: "#247b75",
                  },
                  {
                    name: "Uniform validation MSE",
                    points: run.history
                      .filter((r) => r.validation_loss != null)
                      .map((r) => ({ x: r.step, y: r.validation_loss! })),
                    color: "#cc7040",
                  },
                ]}
              />
              <details>
                <summary>Separate objective components</summary>
                <pre>
                  {JSON.stringify(run.history.at(-1)?.components, null, 2)}
                </pre>
              </details>
            </>
          )}
          {preview && (
            <details open>
              <summary>
                Current-stage validation prediction · update{" "}
                {preview.checkpoint_update}
              </summary>
              <p>
                {preview.preview_label.family} {preview.preview_label.quantity}{" "}
                · physical t={preview.preview_label.physical_time} · RMSE{" "}
                {preview.preview_label.rmse.toPrecision(4)}.{" "}
                {preview.preview_label.scope}. This is a fixed validation
                example, not a test case or a reference substituted for a
                prediction.
              </p>
              <div className="wave-preview-grid">
                <HeatCanvas
                  values={preview.reference}
                  range={[preview.vmin, preview.vmax]}
                  label="Validation reference"
                />
                <HeatCanvas
                  values={preview.values}
                  range={[preview.vmin, preview.vmax]}
                  label="Actual current-stage prediction"
                />
              </div>
            </details>
          )}
          {["queued", "running", "stopping"].includes(run.state) && (
            <button
              onClick={() =>
                void act(async () => {
                  await api(`/training/runs/${run.id}/stop`, {});
                })
              }
            >
              Stop and save
            </button>
          )}
          {run.can_resume && (
            <button
              onClick={() =>
                void act(async () => {
                  await api(`/training/runs/${run.id}/resume`, {});
                })
              }
            >
              Resume recorded run
            </button>
          )}
          {run.model_id && (
            <button
              onClick={() =>
                void act(async () => {
                  const selection = await api<TrainingSelection>(
                    "/training/use",
                    {
                      id: run.model_id,
                      problem,
                      adopt_training_problem: false,
                    },
                  );
                  onUse?.(selection);
                })
              }
            >
              Use this model
            </button>
          )}
          <p>
            Stop preserves optimizer/scaler/RNG and successful-update counters.
            Resume retains the recorded architecture, observations and
            objective; changed science requires a new experiment.
          </p>
        </>
      )}
      <details>
        <summary>Legacy baseline and comparison</summary>
        <p>
          The original factorized AE → flow → evolution models and checkpoints
          retain their original architecture and loader.
        </p>
        <button onClick={onLegacy}>Open legacy training controls</button>
        <p>
          Colab exposes the full reproducible M1–M5 suite, explicit
          start/remaining-run resume and comparison. Substantial suite training
          never starts when opening this screen.
        </p>
      </details>
    </section>
  );
}
