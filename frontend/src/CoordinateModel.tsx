import type { MathematicalRequest } from "./mathematics";
import { useEffect, useState } from "react";
import { api, saveFile } from "./types";
import type { Problem, Row, Slice, Job } from "./types";
import { Button } from "./components/ui/button";
import { FieldPlot } from "./FieldPlot";
type StageRun = {
  id: string;
  stage: string;
  state: string;
  available: boolean;
  model_id?: string;
  progress: number;
  budget: number;
  parameters: number;
  device: string;
  path: string;
  resume_command: string;
  config: { reference_shape: number[]; times: number[] };
  capabilities?: unknown;
  history?: unknown[];
  error?: string;
};
export function CoordinateTraining() {
  const [runs, setRuns] = useState<StageRun[]>([]),
    [error, setError] = useState("");
  useEffect(() => {
    let live = true;
    const refresh = () =>
      void api<StageRun[]>("/coordinate/runs")
        .then((r) => {
          if (live) setRuns(r);
        })
        .catch((e) => {
          if (live) setError(String(e));
        });
    refresh();
    const id = setInterval(refresh, 5000);
    return () => {
      live = false;
      clearInterval(id);
    };
  }, []);
  return (
    <details className="coordinate-training">
      <summary>Latent coordinate heat specialist · two training stages</summary>
      <p>
        Heat only, one spatial dimension per checkpoint, periodic Cartesian
        domain and known constant diffusivity. Autoencoder reconstruction
        supervision precedes frozen-latent flow matching. Query flexibility is
        not a fine-scale accuracy claim. This optional architecture is
        explicitly trained; train-all retains its documented fixed-grid
        catalogue.
      </p>
      <pre>
        pdeflow coordinate-train --config configs/coordinate/heat-smoke.json
        --stage autoencoder --output outputs/local/coordinate/heat-ae{`\n`}
        pdeflow coordinate-train --config configs/coordinate/heat-smoke.json
        --stage flow --autoencoder
        outputs/local/coordinate/heat-ae/autoencoder.pt --output
        outputs/local/coordinate/heat-flow
      </pre>
      <table aria-label="Coordinate training stages">
        <thead>
          <tr>
            <th>Run / stage</th>
            <th>Status / progress</th>
            <th>Reference layout / parameters</th>
            <th>Commands and metadata</th>
          </tr>
        </thead>
        <tbody>
          {runs.map((r) => (
            <tr key={r.id}>
              <td>
                {r.id} / {r.stage}
              </td>
              <td>
                {r.state} · {r.progress}/{r.budget} · export{" "}
                {r.available ? "available" : "missing"}
              </td>
              <td>
                {r.config?.reference_shape.join(" × ")} · {r.parameters}{" "}
                parameters · {r.device}
              </td>
              <td>
                <details>
                  <summary>Inspect stage, loss history and resume</summary>
                  <pre>
                    {r.resume_command}
                    {"\n"}
                    {JSON.stringify(r, null, 2)}
                  </pre>
                </details>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      {!runs.length && <p>No coordinate stages trained in this workspace.</p>}
      {error && <p role="alert">{error}</p>}
    </details>
  );
}
function readSettings() {
  try {
    return JSON.parse(
      localStorage.getItem("pdeflow.coordinate.settings") || "{}",
    );
  } catch {
    return {};
  }
}
export function CoordinateReconstruction({
  problem,
  rows,
  referenceId,
  noiseStd,
  ready,
  mathematicalRequest,
  selectedModel,
}: {
  problem: Problem;
  rows: Row[];
  referenceId: string | null;
  noiseStd: number;
  ready: boolean;
  mathematicalRequest: MathematicalRequest;
  selectedModel?: string;
}) {
  const [saved] = useState(readSettings);
  const [models, setModels] = useState<StageRun[]>([]),
    [model, setModel] = useState<string>(String(saved.model ?? "")),
    [shape, setShape] = useState(saved.shape ?? ""),
    [queries, setQueries] = useState(saved.queries ?? ""),
    [observationText, setObservationText] = useState(
      saved.observationText ?? "",
    ),
    [steps, setSteps] = useState(saved.steps ?? 8),
    [particles, setParticles] = useState(saved.particles ?? 2),
    [seed, setSeed] = useState(saved.seed ?? 91),
    [chunk, setChunk] = useState(saved.chunk ?? 256),
    [device, setDevice] = useState<string>(saved.device ?? "cpu"),
    [error, setError] = useState(""),
    [busy, setBusy] = useState(false),
    [job, setJob] = useState<Job | null>(() => {
      try {
        return JSON.parse(
          localStorage.getItem("pdeflow.coordinate.job") || "null",
        );
      } catch {
        return null;
      }
    }),
    [report, setReport] = useState<unknown>(null),
    [slice, setSlice] = useState<Slice | null>(null),
    [time, setTime] = useState(0),
    [axis, setAxis] = useState(2),
    [index, setIndex] = useState(0),
    [statistic, setStatistic] = useState("mean");
  useEffect(() => {
    try {
      localStorage.setItem(
        "pdeflow.coordinate.settings",
        JSON.stringify({
          model,
          shape,
          queries,
          observationText,
          steps,
          particles,
          seed,
          chunk,
          device,
        }),
      );
    } catch {
      /* No scientific artifact is changed by draft quota failures. */
    }
  }, [model, shape, queries, observationText, steps, particles, seed, chunk, device]);
  function refresh() {
    void api<StageRun[]>("/coordinate/runs")
      .then((r) => {
        const m = r.filter((x) => x.stage === "flow" && x.available);
        setModels(m);
        setModel((old) =>
          selectedModel && m.some((x) => x.model_id === selectedModel) ? selectedModel : m.some((x) => x.model_id === old) ? old : (m[0]?.model_id ?? ""),
        );
      })
      .catch((e) => setError(String(e)));
  }
  useEffect(() => {
    refresh();
  // A model is selected only by an explicit Use action; discovery remains passive.
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedModel]);
  useEffect(() => {
    localStorage.setItem("pdeflow.coordinate.job", JSON.stringify(job));
    if (!job || !["queued", "running"].includes(job.status)) return;
    let live = true;
    const timer = setInterval(
      () =>
        void api<Job>(`/jobs/${job.id}`)
          .then((j) => {
            if (live) setJob(j);
          })
          .catch((e) => {
            if (live) setError(String(e));
          }),
      600,
    );
    return () => {
      live = false;
      clearInterval(timer);
    };
  }, [job]);
  useEffect(() => {
    if (job?.status !== "succeeded") return;
    let live = true;
    void api(`/coordinate/jobs/${job.id}/result`)
      .then((r) => {
        if (live) setReport(r);
      })
      .catch((e) => {
        if (live) setError(String(e));
      });
    void api<Slice>(
      `/coordinate/jobs/${job.id}/slice?time_index=${time}&axis=${axis}&index=${index}&statistic=${statistic}`,
    )
      .then((s) => {
        if (live) setSlice(s);
      })
      .catch((e) => {
        if (live) {
          setSlice(null);
          setError(String(e));
        }
      });
    return () => {
      live = false;
    };
  }, [job, time, axis, index, statistic]);
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
  const viewed =
    (report as { metrics?: { physical_problem?: Problem } } | null)?.metrics
      ?.physical_problem ?? problem;
  function request() {
    const p = structuredClone(problem);
    if (shape.trim()) p.grid.shape = shape.split(",").map(Number);
    const observationRows = (
      observationText.trim() ? JSON.parse(observationText) : rows
    ).map((r: Row) => ({ ...r, sigma: r.sigma ?? noiseStd }));
    const mathematical = structuredClone(mathematicalRequest);
    mathematical.measurements = {
      ...mathematical.measurements,
      rows: observationRows,
      status: "known",
    };
    mathematical.observation_operator = {
      kind: "coordinate_point_set",
      status: "known",
      details:
        "Scalar physical-coordinate sensor set for the explicit coordinate specialist",
    };
    return {
      mathematical_request: mathematical,
      model_id: model,
      problem: p,
      rows: observationRows,
      queries: queries.trim() ? JSON.parse(queries) : null,
      steps,
      particles,
      seed,
      chunk_size: chunk,
      device,
    };
  }
  return (
    <section className="coordinate-reconstruction">
      <h3>Resolution-flexible heat reconstruction</h3>
      <p>
        Observation count controls measurement data. Number of reconstructions
        controls generated samples. This model learns observation-set
        conditioning; it does not use the complete reference or the fixed-grid
        model’s guidance controller. Flow time s∈[0,1] is separate from physical
        time t.
      </p>
      <Button variant="outline" onClick={refresh}>
        Refresh coordinate checkpoints
      </Button>
      <label>
        Coordinate flow checkpoint
        <select
          aria-label="Coordinate checkpoint"
          value={model}
          onChange={(e) => setModel(e.target.value)}
        >
          <option value="">Choose completed latent flow</option>
          {models.map((m) => (
            <option key={m.id} value={m.model_id}>
              {m.id} · {m.progress} updates
            </option>
          ))}
        </select>
      </label>
      {!models.length && (
        <p>
          Missing checkpoint: complete autoencoder and flow stages in Models &
          Training. No training starts here.
        </p>
      )}
      <div className="preview-controls">
        <label>
          Coordinate compute device (float32)
          <select
            aria-label="Coordinate compute device"
            value={device}
            onChange={(e) => setDevice(e.target.value)}
          >
            {["cpu", "mps", "cuda", "auto"].map((d) => (
              <option key={d} value={d}>
                {d}
              </option>
            ))}
          </select>
        </label>
        <label>
          Query grid nodes, comma separated
          <input
            aria-label="Coordinate query grid"
            placeholder={problem.grid.shape.join(",")}
            value={shape}
            onChange={(e) => setShape(e.target.value)}
          />
        </label>
        <label>
          Flow integration steps
          <input
            aria-label="Coordinate flow steps"
            type="number"
            min={1}
            value={steps}
            onChange={(e) => setSteps(Number(e.target.value))}
          />
        </label>
        <label>
          Number of reconstructions
          <input
            aria-label="Coordinate reconstructions"
            type="number"
            min={1}
            max={16}
            value={particles}
            onChange={(e) => setParticles(Number(e.target.value))}
          />
        </label>
        <label>
          Reconstruction seed
          <input
            aria-label="Coordinate seed"
            type="number"
            value={seed}
            onChange={(e) => setSeed(Number(e.target.value))}
          />
        </label>
        <label>
          Query chunk size
          <input
            aria-label="Coordinate chunk size"
            type="number"
            min={1}
            value={chunk}
            onChange={(e) => setChunk(Number(e.target.value))}
          />
        </label>
      </div>
      <details>
        <summary>Scattered measurements or zero-observation prior</summary>
        <p>
          Optional scalar rows with field=u, x/y/z, physical time, observed
          value, absolute sigma. Blank uses the current measurements; []
          explicitly requests the trained prior. These points may be off the
          fixed-grid model’s nodes. Missing is not measured zero.
        </p>
        <textarea
          aria-label="Coordinate observation rows"
          value={observationText}
          onChange={(e) => setObservationText(e.target.value)}
        />
      </details>
      <details>
        <summary>Scattered physical queries</summary>
        <p>
          Optional JSON rows [x,(y,z),physical time]. Preserved query order, no
          extrapolation. Leave blank for a regular grid.
        </p>
        <textarea
          aria-label="Scattered coordinate queries"
          value={queries}
          onChange={(e) => setQueries(e.target.value)}
        />
      </details>
      <p>
        {rows.length} scalar measurements;{" "}
        {referenceId
          ? "reference available for post-run evaluation"
          : "measurements only: truth error unavailable; physics diagnostics remain available"}
        . Zero measurements use an explicitly trained prior. New grids do not
        establish finer learned physics.
      </p>
      {!ready && (
        <p role="alert">
          Current equation or declarations are unsupported or still parsing.
          Resolve the current draft before coordinate reconstruction.
        </p>
      )}
      <Button
        disabled={!ready || !model || busy}
        onClick={() =>
          void act(async () => {
            await api("/coordinate/validate", request());
            setJob(await api<Job>("/coordinate/reconstruct", request()));
            setReport(null);
            setSlice(null);
          })
        }
      >
        Reconstruct with coordinate specialist
      </Button>
      {job && (
        <p role="status">
          Coordinate job {job.id}: {job.status} · {job.completed_steps ?? 0}/
          {job.total_steps ?? steps}
        </p>
      )}
      {job?.status === "failed" && <p role="alert">{job.error?.message}</p>}
      {job?.status === "succeeded" && (
        <>
          <Button
            variant="outline"
            disabled={!referenceId || busy}
            onClick={() =>
              void act(async () => {
                await api(`/coordinate/jobs/${job.id}/compare`, {
                  reference_id: referenceId,
                });
                setReport(await api(`/coordinate/jobs/${job.id}/result`));
                setSlice(
                  await api<Slice>(
                    `/coordinate/jobs/${job.id}/slice?time_index=${time}&axis=${axis}&index=${index}&statistic=${statistic}`,
                  ),
                );
              })
            }
          >
            Compare coordinate result with reference
          </Button>
          <label>
            Physical time index
            <input
              aria-label="Coordinate physical time"
              type="range"
              min={0}
              max={viewed.grid.times.length - 1}
              value={time}
              onChange={(e) => setTime(Number(e.target.value))}
            />
          </label>
          {viewed.grid.shape.length === 3 && (
            <>
              <label>
                Slice axis
                <select
                  aria-label="Coordinate slice axis"
                  value={axis}
                  onChange={(e) => setAxis(Number(e.target.value))}
                >
                  {["x", "y", "z"].map((a, i) => (
                    <option key={a} value={i}>
                      {a}
                    </option>
                  ))}
                </select>
              </label>
              <label>
                Slice index
                <input
                  aria-label="Coordinate slice index"
                  type="number"
                  value={index}
                  min={0}
                  onChange={(e) => setIndex(Number(e.target.value))}
                />
              </label>
            </>
          )}
          <label>
            Displayed reconstruction statistic
            <select
              aria-label="Coordinate statistic"
              value={statistic}
              onChange={(e) => setStatistic(e.target.value)}
            >
              <option value="mean">Ensemble mean</option>
              <option value="spread">Empirical spread (not calibrated)</option>
              {Array.from({ length: particles }, (_, i) => (
                <option key={i} value={`sample:${i}`}>
                  Sample {i + 1}
                </option>
              ))}
            </select>
          </label>
          {slice && (
            <>
              <p>
                {slice.observations.length} displayed measurements at this
                physical time/slice; {slice.off_slice_observation_count ?? 0}{" "}
                off this view. Reference and reconstruction share a color scale.
              </p>
              <div className="plot-grid">
                <FieldPlot
                  slice={slice}
                  values={slice.reference}
                  title="Reference"
                  observations
                />
                <FieldPlot
                  slice={slice}
                  values={slice.values}
                  title="Coordinate reconstruction"
                  observations
                />
                <FieldPlot
                  slice={slice}
                  values={slice.error}
                  title="Absolute error"
                />
              </div>
            </>
          )}
          <details>
            <summary>Coordinate metrics and ordered query table</summary>
            <pre>{JSON.stringify(report, null, 2)}</pre>
          </details>
          <Button
            variant="outline"
            onClick={() => saveFile("coordinate-report.json", report)}
          >
            Export coordinate report
          </Button>
          <a href={`/api/coordinate/jobs/${job.id}/artifacts/result.npz`}>
            Export ordered queries and samples
          </a>
        </>
      )}
      {busy && <p role="status">Validating coordinate request…</p>}
      {error && <p role="alert">{error}</p>}
    </section>
  );
}
