import { CoordinateTraining } from "./CoordinateModel";
import { TrainModel } from "./TrainModel";
import type { TrainingSelection } from "./TrainModel";
import { TrainingOverview } from "./TrainingOverview";
import { useEffect, useState } from "react";
import { api } from "./types";
import type { Model, Problem } from "./types";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
  CardDescription,
} from "./components/ui/card";
import { Button } from "./components/ui/button";
import { LineChart } from "./ScienceCharts";
type TrainingRun = {
  id: string;
  run_id: string;
  name: string | null;
  specialist: string;
  preset: string;
  profile: string;
  state: string;
  progress: number;
  budget: number;
  device: string;
  grid: { shape: number[]; times: number[] };
  history: { step: number; training_loss: number; validation_loss?: number }[];
  checkpoint_positions: number[];
  history_total: number;
  downsampled: boolean;
  meaning: string;
  resume_command: string;
};
type TrainingPreset = {
  name: string;
  grid: number[];
  times: number[];
  steps: number;
  batch_size: number;
  profile: string;
  intended_device: string;
};
export function ModelsTraining({
  models,
  onRefresh,
  problem,
  onUse,
}: {
  models: Model[];
  problem?: Problem;
  onRefresh: () => void;
  onUse?: (selection: TrainingSelection) => void;
}) {
  const [suites, setSuites] = useState<
    {
      id: string;
      state: string;
      path: string;
      resume_command: string;
      entries: {
        preset: string;
        state: string;
        run_id: string | null;
        latest_update: number | null;
        run_path: string;
      }[];
    }[]
  >([]);
  const [presets, setPresets] = useState<TrainingPreset[]>([]);
  const [runs, setRuns] = useState<TrainingRun[]>([]),
    [error, setError] = useState(""),
    [selected, setSelected] = useState("");
  const [advancedOpen, setAdvancedOpen] = useState(false);
  useEffect(() => {
    void api<typeof suites>("/training-suites")
      .then(setSuites)
      .catch((e) => setError(String(e)));
  }, [models]);
  useEffect(() => {
    void api<TrainingPreset[]>("/training-presets")
      .then(setPresets)
      .catch((e) => setError(String(e)));
  }, []);
  const refresh = () => {
    onRefresh();
    void api<TrainingRun[]>("/training-runs")
      .then((v) => {
        setRuns(v);
        setSelected((s) => (v.some((r) => r.id === s) ? s : (v[0]?.id ?? "")));
        setError("");
      })
      .catch((e) => setError(String(e)));
  };
  useEffect(() => {
    void api<TrainingRun[]>("/training-runs")
      .then((v) => {
        setRuns(v);
        setSelected(v[0]?.id ?? "");
      })
      .catch((e) => setError(String(e)));
  }, []);
  const run = runs.find((r) => r.id === selected);
  return (
    <div className="models-training">
      <TrainModel problem={problem} onUse={onUse} onRefresh={onRefresh} />
      <details className="advanced-training-tools" onToggle={e => setAdvancedOpen(e.currentTarget.open)}>
      <summary>Advanced tools, legacy training stages and storage</summary>
      {advancedOpen && <>
      <CoordinateTraining />
      <div className="section-intro">
        <h1>Models & Training</h1>
        <p>
          Models are specialists for a particular equation and layout. Finishing
          training is not scientific validation.
        </p>
        <Button variant="outline" onClick={refresh}>
          Refresh models & history
        </Button>
      </div>
      <TrainingOverview problem={problem} />
      <div className="model-list">
        {models.length ? (
          models.map((m) => (
            <Card key={m.id}>
              <CardHeader>
                <CardTitle>
                  {m.run_name || m.problem.family} ·{" "}
                  {m.problem.grid.shape.length}D
                </CardTitle>
                <CardDescription>{m.training.status}</CardDescription>
              </CardHeader>
              <CardContent>
                <p>
                  {m.problem.grid.shape.join(" × ")} ·{" "}
                  {m.problem.grid.temporal_kind === "static"
                    ? "static fields"
                    : `${m.problem.grid.times.length} saved physical times`}{" "}
                  · {m.training.updates} updates
                </p>
                <p>{m.name}</p>
                {m.architecture && (
                  <p>
                    {m.architecture.configuration.name} ·{" "}
                    {m.architecture.parameters.toLocaleString()} parameters ·
                    width {m.architecture.configuration.width},{" "}
                    {m.architecture.configuration.blocks} blocks/level,{" "}
                    {m.architecture.configuration.levels} levels ·{" "}
                    {m.architecture.deepest_path_3x3_layers} spatial 3×3 layers
                    on deepest path. {m.architecture.resolution_policy}
                  </p>
                )}
                <p>
                  Exact equation, fields, boundary and layout matching are
                  required. Training ranges not recorded in the export remain
                  unknown.
                </p>
              </CardContent>
            </Card>
          ))
        ) : (
          <Card>
            <CardHeader>
              <CardTitle>No trained models yet</CardTitle>
            </CardHeader>
            <CardContent>
              <p>
                Start with a real bounded check, or choose a longer pilot below.
                Startup never trains automatically.
              </p>
              <code>
                pdeflow train --preset quickstart-heat-2d --device cpu
              </code>
            </CardContent>
          </Card>
        )}
      </div>
      <Card>
        <CardHeader>
          <CardTitle>Sequential training suites</CardTitle>
          <CardDescription>
            Versioned full catalogue and frozen suite snapshots; opening this
            page never starts training.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="command-sheet">
            <code>pdeflow train-all --plan</code>
            <code>pdeflow train-all --suite quickstart --device cpu</code>
            <code>pdeflow train-all --device mps</code>
            <code>pdeflow resume-suite latest</code>
            <code>pdeflow suite-status latest</code>
          </div>
          <p>
            Local pilot: 3000 updates per 2D child, 1500 per 3D child, batch 2.
            Quickstart: identical explicit grids, 2 updates each, batch 1. Heat,
            Darcy and chemistry execute sequentially. Checkpoint availability
            does not establish scientific quality.
          </p>
          {suites.map((s) => (
            <details key={s.id}>
              <summary>
                {s.id} · {s.state}
              </summary>
              <p>{s.path}</p>
              <code>{s.resume_command}</code>
              <table aria-label={`Suite ${s.id}`}>
                <thead>
                  <tr>
                    <th>Preset</th>
                    <th>Status / update</th>
                    <th>Child run</th>
                  </tr>
                </thead>
                <tbody>
                  {s.entries.map((e) => (
                    <tr key={e.preset}>
                      <td>{e.preset}</td>
                      <td>
                        {e.state} / {e.latest_update ?? "pending"}
                      </td>
                      <td title={e.run_path}>
                        <Button
                          variant="ghost"
                          disabled={!runs.some((r) => r.run_id === e.run_id)}
                          onClick={() => {
                            const child = runs.find(
                              (r) => r.run_id === e.run_id,
                            );
                            if (child) {
                              setSelected(child.id);
                              document
                                .getElementById("training-history")
                                ?.scrollIntoView({ behavior: "smooth" });
                            }
                          }}
                        >
                          {e.run_id ?? e.run_path}
                        </Button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </details>
          ))}
          {!suites.length && (
            <p>No training suites recorded in this workspace.</p>
          )}
        </CardContent>
      </Card>
      <Card>
        <CardHeader>
          <CardTitle>Train a specialist</CardTitle>
          <CardDescription>
            The package prepares fixed evaluation and calibration, generates
            training problems online and publishes checkpoints automatically.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="command-sheet">
            {["heat-2d", "darcy-2d", "chemistry-2d"].map((name) => (
              <div key={name}>
                <strong>{name}</strong>
                <code>pdeflow train --preset {name} --device mps</code>
              </div>
            ))}
          </div>
          <details>
            <summary>All named presets and actual budgets</summary>
            <div className="bounded-table">
              <table aria-label="Available training presets">
                <thead>
                  <tr>
                    <th>Preset</th>
                    <th>Grid / physical times</th>
                    <th>Updates / batch</th>
                    <th>Status</th>
                  </tr>
                </thead>
                <tbody>
                  {presets.map((p) => (
                    <tr key={p.name}>
                      <td>{p.name}</td>
                      <td>
                        {p.grid.join(" × ")} / {p.times.length}
                      </td>
                      <td>
                        {p.steps} / {p.batch_size}
                      </td>
                      <td>
                        {p.profile} · {p.intended_device}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <p>
              These are resolved training configurations, not measured duration
              or accuracy guarantees. No training starts from this page.
            </p>
          </details>
          <p>
            Use --device cpu explicitly if MPS is unavailable. Full pilot
            durations and scientific quality are unestablished.
          </p>
          <p>
            <code>pdeflow resume latest</code> ·{" "}
            <code>pdeflow evaluate --run latest</code> ·{" "}
            <code>pdeflow doctor</code>
          </p>
        </CardContent>
      </Card>
      <Card>
        <CardHeader>
          <CardTitle>
            <span id="training-history">Recorded training history</span>
          </CardTitle>
          <CardDescription>
            Actual optimizer updates and fixed-validation objectives. No
            smoothing; this is generative training loss, not reconstruction
            error.
          </CardDescription>
        </CardHeader>
        <CardContent>
          {error && <p role="alert">{error}</p>}
          {runs.length ? (
            <>
              <label>
                Managed training run
                <select
                  aria-label="Training history run"
                  value={selected}
                  onChange={(e) => setSelected(e.target.value)}
                >
                  {runs.map((r) => (
                    <option key={r.id} value={r.id}>
                      {r.name || r.run_id} · {r.specialist} · {r.progress}/
                      {r.budget}
                    </option>
                  ))}
                </select>
              </label>
              {run && (
                <>
                  <p>
                    {run.state} · {run.device} · {run.grid.shape.join(" × ")} ·{" "}
                    {run.profile}
                  </p>
                  <LineChart
                    title="Training and validation loss"
                    xlabel="Optimizer update"
                    ylabel="CFM objective"
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
                        name: "Validation / checkpoint",
                        color: "#aa6b38",
                        dashed: true,
                        points: run.history
                          .filter((h) => h.validation_loss !== undefined)
                          .map((h) => ({ x: h.step, y: h.validation_loss! })),
                      },
                    ]}
                  />
                  <p>
                    {run.meaning}{" "}
                    {run.downsampled
                      ? `Bounded extrema-preserving display of ${run.history_total} rows; no smoothing.`
                      : "All recorded rows shown."}
                  </p>
                  <p>
                    Saved checkpoint updates:{" "}
                    {run.checkpoint_positions?.join(", ") ||
                      "No checkpoint events recorded"}
                    . Validation points remain separate from training loss.
                  </p>
                  <code>{run.resume_command}</code>
                </>
              )}
            </>
          ) : (
            <p>
              No managed training history is recorded in this workspace. Linked
              external checkpoints do not invent a training curve.
            </p>
          )}
        </CardContent>
      </Card>
      </>}
      </details>
    </div>
  );
}
