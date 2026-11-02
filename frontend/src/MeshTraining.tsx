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
type Plan = {
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
        capacity,
        batch_size: batch,
        microbatch: micro,
        field_loss: loss,
        ...JSON.parse(advanced || "{}"),
      },
    }),
    [family, model, budget, device, capacity, batch, micro, loss, advanced],
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
            onChange={(e) => {
              setBudget(e.target.value);
              if (e.target.value === "pipeline-check") {
                setCapacity("pipeline-check");
                setBatch(2);
                setMicro(2);
              }
            }}
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
      <details>
        <summary>Advanced model and batch settings</summary>
        <div className="number-grid">
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
            Forward microbatch
            <input
              aria-label="Mesh forward microbatch"
              type="number"
              min={1}
              value={micro}
              onChange={(e) => setMicro(e.target.valueAsNumber)}
            />
          </label>
        </div>
        <p>
          {Math.ceil(batch / micro)} forward/backward passes, summing
          per-example losses over the effective batch. Model-specific noise,
          objective weights, corruption and samplers are optional JSON
          overrides.
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
