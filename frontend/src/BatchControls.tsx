import { useEffect, useState } from "react";
import { api } from "./types";
export type BatchStage = {
  label: string;
  microbatch: number;
  requested_microbatch: number | "auto";
  accumulation: number;
  effective_batch: number;
  last_microbatch: number;
  planned_successful_updates: number;
  approximate_examples: number;
};
type Row = {
  stage: string;
  microbatch: number;
  effective_batch: number;
  accumulation: number;
  status: string;
  cache: string;
  examples_per_second?: number;
  median_seconds?: number;
  peak_allocated?: number;
  skipped_updates?: number;
  error?: string;
  reason?: string;
};
type Job = {
  id: string;
  state: string;
  rows: Row[];
  recommendation: Record<
    string,
    { microbatch: number; effective_batch: number; reason: string }
  >;
  plan?: { estimate: { seconds: number | null; meaning?: string } };
  error?: string;
};
const stages = [
  ["autoencoder", "A1 representation"],
  ["evolution", "A2 evolution"],
  ["flow", "B conditional flow"],
];
export function BatchControls({
  request,
  batches,
  onApply,
  onDirty,
  onRunning,
  active = false,
}: {
  request: Record<string, unknown>;
  batches: Record<string, BatchStage>;
  onApply: (a: Record<string, unknown>) => void;
  onDirty: (dirty: boolean) => void;
  onRunning?: (running: boolean) => void;
  active?: boolean;
}) {
  const [effective, setEffective] = useState(
    String(batches.autoencoder.effective_batch),
  );
  const [microMode, setMicroMode] = useState("auto");
  const [micro, setMicro] = useState("8");
  const [overrides, setOverrides] = useState<Record<string, string>>({});
  const [policy, setPolicy] = useState("keep_updates");
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [mode, setMode] = useState("efficiency"),
    [candidateMode, setCandidateMode] = useState("auto"),
    [candidates, setCandidates] = useState("8,16,32"),
    [selectedStages, setSelectedStages] = useState(["autoencoder"]);
  const [learningBatches, setLearningBatches] = useState("8,16"),
    [learningPolicy, setLearningPolicy] = useState("examples"),
    [learningBudget, setLearningBudget] = useState("256");
  const [cap, setCap] = useState("120");
  const [estimate, setEstimate] = useState<{
    key: string;
    text: string;
  } | null>(null);
  const [job, setJob] = useState<Job | null>(null);
  const [saved, setSaved] = useState<
    { id: string; state: string; mode: string }[]
  >([]);
  const [appliedExperimentId, setAppliedExperimentId] = useState<string | null>(
    null,
  );
  useEffect(() => {
    void api<{ id: string; state: string; mode: string }[]>(
      "/training/batch/benchmarks",
    )
      .then((rows) => setSaved(Array.isArray(rows) ? rows : []))
      .catch((e) => setError(String(e)));
  }, []);
  const [chosen, setChosen] = useState<Record<string, string>>({});
  const [application, setApplication] = useState<{
    before: Record<string, BatchStage>;
    after: Record<string, BatchStage>;
  } | null>(null);
  const signature = JSON.stringify(batches);
  const [previousSignature, setPreviousSignature] = useState(signature);
  if (previousSignature !== signature) {
    setPreviousSignature(signature);
    const advanced = request.advanced as Record<string, unknown> | undefined;
    const training = advanced?.training_options as
      | {
          microbatch?: number | string;
          stage_batches?: Record<string, { microbatch: number | string }>;
        }
      | undefined;
    setEffective(String(batches.autoencoder.effective_batch));
    setMicroMode(typeof training?.microbatch === "number" ? "manual" : "auto");
    setMicro(String(training?.microbatch || batches.autoencoder.microbatch));
    setOverrides(
      Object.fromEntries(
        Object.entries(training?.stage_batches || {}).map(([k, v]) => [
          k,
          String(v.microbatch),
        ]),
      ),
    );
  }
  useEffect(() => {
    onDirty(false);
  }, [signature, onDirty]);
  const running =
    !!job && ["queued", "running", "stopping"].includes(job.state);
  useEffect(() => {
    onRunning?.(running || busy);
  }, [running, busy, onRunning]);
  useEffect(() => {
    if (!job || !running) return;
    let live = true;
    const timer = setInterval(() => {
      void api<Job>(`/training/batch/benchmark/${job.id}`)
        .then((r) => {
          if (live) setJob(r);
        })
        .catch((e) => setError(String(e)));
    }, 700);
    return () => {
      live = false;
      clearInterval(timer);
    };
  }, [job, running]);
  function edit(fn: () => void) {
    fn();
    setAppliedExperimentId(null);
    onDirty(true);
    setMessage("Batch edits are not applied yet.");
  }
  const options = {
    mode,
    stages: selectedStages,
    candidates:
      candidateMode === "auto"
        ? "auto"
        : candidates
            .split(/[,\s]+/)
            .filter(Boolean)
            .map(Number),
    effective_batch: Number(effective),
    effective_batches: learningBatches
      .split(/[,\s]+/)
      .filter(Boolean)
      .map(Number),
    budget_policy: learningPolicy,
    budget: Number(learningBudget),
    max_seconds: Number(cap),
    warmup: 2,
    measured: 5,
  };
  const key = JSON.stringify({ request, options });
  async function action(fn: () => Promise<void>) {
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
  async function apply(source: "manual" | "recommended" | "selected") {
    await action(async () => {
      const result = await api<{
        request: { advanced: Record<string, unknown> };
        application: {
          before: Record<string, BatchStage>;
          after: Record<string, BatchStage>;
        };
        state: string;
      }>("/training/batch/apply", {
        request,
        budget_policy: policy,
        ...(source === "manual"
          ? {
              settings: {
                effective_batch: Number(effective),
                microbatch: microMode === "auto" ? "auto" : Number(micro),
                stages: Object.fromEntries(
                  Object.entries(overrides)
                    .filter(([, v]) => v)
                    .map(([s, v]) => [
                      s,
                      { microbatch: v === "auto" ? "auto" : Number(v) },
                    ]),
                ),
              },
            }
          : {
              experiment_id: job?.id,
              recommended: source === "recommended",
              selections: Object.fromEntries(
                Object.entries(chosen)
                  .filter(([, v]) => v)
                  .map(([s, v]) => {
                    const [e, m] = v.split(":").map(Number);
                    return [s, { effective_batch: e, microbatch: m }];
                  }),
              ),
            }),
      });
      setApplication(result.application);
      setAppliedExperimentId(source === "manual" ? null : job?.id || null);
      onApply(result.request.advanced);
      onDirty(false);
      setMessage(result.state);
    });
  }
  return (
    <section
      aria-label="Batch settings"
      className="space-y-3 rounded border p-3"
    >
      <p>
        <strong>Batch settings</strong>
      </p>
      <div className="grid gap-3 md:grid-cols-2">
        <label>
          Effective batch size
          <input
            aria-label="Effective batch size"
            type="number"
            min="1"
            value={effective}
            onChange={(e) => edit(() => setEffective(e.target.value))}
          />
        </label>
        <label>
          GPU microbatch
          <select
            aria-label="GPU microbatch"
            value={microMode}
            onChange={(e) => edit(() => setMicroMode(e.target.value))}
          >
            <option value="auto">Auto — conservative, unqualified</option>
            <option value="manual">Manual</option>
          </select>
        </label>
        {microMode === "manual" && (
          <label>
            Manual GPU microbatch
            <input
              aria-label="Manual GPU microbatch"
              type="number"
              min="1"
              list="batch-sizes"
              value={micro}
              onChange={(e) => edit(() => setMicro(e.target.value))}
            />
          </label>
        )}
      </div>
      <datalist id="batch-sizes">
        {[8, 16, 32, 64, 128, 256, 512, 1024].map((n) => (
          <option key={n} value={n} />
        ))}
      </datalist>
      <p className="text-xs">
        GPU microbatch = examples in one neural pass. Gradient accumulation =
        passes per optimizer update. Effective batch = examples per update. Auto
        uses min(effective batch, 8) until measured; it is not hardware
        qualification. Custom positive sizes, including 1024, require fit
        checks.
      </p>
      <label>
        Budget when effective batch changes
        <select
          aria-label="Batch budget policy"
          value={policy}
          onChange={(e) => setPolicy(e.target.value)}
        >
          <option value="keep_updates">
            Keep successful update count — changes examples processed
          </option>
          <option value="keep_examples">
            Keep approximate examples — rounds updates up
          </option>
        </select>
      </label>
      <p className="text-xs">
        Learning rate stays unchanged. Neither policy guarantees the same
        learning outcome.
      </p>
      <details>
        <summary>Advanced per-stage GPU microbatches</summary>
        {stages.map(([s, label]) => (
          <label key={s}>
            {label}
            <input
              aria-label={`${label} GPU microbatch`}
              type="text"
              placeholder="Inherit / auto / positive size"
              value={overrides[s] || ""}
              onChange={(e) =>
                edit(() => setOverrides({ ...overrides, [s]: e.target.value }))
              }
            />
          </label>
        ))}
      </details>
      <button
        type="button"
        disabled={active || busy || running}
        onClick={() => void apply("manual")}
      >
        Apply selected batch settings
      </button>
      <p role="status">{message}</p>
      {application && (
        <ul>
          {stages.map(([s, label]) => (
            <li key={s}>
              {label}: GPU microbatch {application.before[s].microbatch} →{" "}
              {application.after[s].microbatch}; accumulation{" "}
              {application.before[s].accumulation} →{" "}
              {application.after[s].accumulation}; effective{" "}
              {application.before[s].effective_batch} →{" "}
              {application.after[s].effective_batch}
            </li>
          ))}
        </ul>
      )}
      <table aria-label="Authoritative batch configuration">
        <thead>
          <tr>
            <th>Stage</th>
            <th>GPU microbatch</th>
            <th>Gradient accumulation</th>
            <th>Effective batch</th>
            <th>Planned examples</th>
          </tr>
        </thead>
        <tbody>
          {stages.map(([s, label]) => (
            <tr key={s}>
              <td>{label}</td>
              <td>{batches[s].microbatch}</td>
              <td>
                {batches[s].accumulation}
                {batches[s].last_microbatch !== batches[s].microbatch
                  ? ` (last ${batches[s].last_microbatch})`
                  : ""}
              </td>
              <td>{batches[s].effective_batch}</td>
              <td>{batches[s].approximate_examples}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <details>
        <summary>Benchmark batch sizes (optional)</summary>
        {saved.length > 0 && (
          <label>
            Saved batch experiment
            <select
              aria-label="Saved batch experiment"
              value={job?.id || ""}
              onChange={(e) =>
                void action(async () => {
                  if (e.target.value)
                    setJob(
                      await api<Job>(
                        `/training/batch/benchmark/${e.target.value}`,
                      ),
                    );
                })
              }
            >
              <option value="">Select saved results</option>
              {saved.map((r) => (
                <option key={r.id} value={r.id}>
                  {r.id} · {r.mode} · {r.state}
                </option>
              ))}
            </select>
          </label>
        )}
        <p>
          Benchmarks disposable configurations. Review results, then apply a
          selection above/below to update the next training submission.
        </p>
        <label>
          Experiment mode
          <select
            aria-label="Batch experiment mode"
            value={mode}
            onChange={(e) => setMode(e.target.value)}
          >
            <option value="efficiency">
              GPU batching efficiency — fixed effective batch
            </option>
            <option value="learning">
              Effective-batch learning comparison
            </option>
          </select>
        </label>
        {stages.map(([s, label]) => (
          <label key={s}>
            <input
              type="checkbox"
              checked={selectedStages.includes(s)}
              onChange={(e) =>
                setSelectedStages(
                  e.target.checked
                    ? [...selectedStages, s]
                    : selectedStages.filter((x) => x !== s),
                )
              }
            />
            {label}
          </label>
        ))}
        {mode === "efficiency" ? (
          <>
            <label>
              Candidate selection
              <select
                aria-label="Batch candidate selection"
                value={candidateMode}
                onChange={(e) => setCandidateMode(e.target.value)}
              >
                <option value="auto">
                  Auto — progressive practical subset
                </option>
                <option value="manual">Editable candidate list</option>
              </select>
            </label>
            {candidateMode === "manual" && (
              <label>
                GPU microbatch candidates
                <input
                  aria-label="GPU microbatch candidates"
                  value={candidates}
                  onChange={(e) => setCandidates(e.target.value)}
                />
              </label>
            )}
          </>
        ) : (
          <>
            <p>
              Changes optimization behaviour. Equal updates are unequal data
              exposure. No automatic recommendation from learning/test
              performance.
            </p>
            <label>
              Effective batch candidates
              <input
                aria-label="Effective batch candidates"
                value={learningBatches}
                onChange={(e) => setLearningBatches(e.target.value)}
              />
            </label>
            <label>
              Learning comparison budget
              <select
                value={learningPolicy}
                onChange={(e) => setLearningPolicy(e.target.value)}
              >
                <option value="examples">Fixed examples processed</option>
                <option value="wall">Fixed wall seconds</option>
                <option value="updates">
                  Fixed successful updates — unequal examples
                </option>
              </select>
              <input
                aria-label="Learning comparison budget"
                type="number"
                min="1"
                value={learningBudget}
                onChange={(e) => setLearningBudget(e.target.value)}
              />
            </label>
          </>
        )}
        <label>
          Maximum experiment seconds
          <input
            aria-label="Maximum experiment seconds"
            type="number"
            min="1"
            value={cap}
            onChange={(e) => setCap(e.target.value)}
          />
        </label>
        <button
          type="button"
          disabled={active || busy || running}
          onClick={() =>
            void action(async () => {
              const p = await api<{
                rows: Row[];
                estimate: {
                  seconds: number | null;
                  runtime_cap_seconds: number;
                  meaning: string;
                };
              }>("/training/batch/estimate", { request, options });
              setEstimate({
                key,
                text: `${p.rows.length} stage/candidate/cache trials. Estimated seconds: ${p.estimate.seconds ?? "unknown until first measured pilot"}; runtime cap ${p.estimate.runtime_cap_seconds}s. ${p.estimate.meaning}`,
              });
            })
          }
        >
          Estimate experiment
        </button>
        {estimate?.key === key && <p>{estimate.text}</p>}
        <button
          type="button"
          disabled={active || busy || running || estimate?.key !== key}
          onClick={() =>
            void action(async () => {
              setJob(
                await api<Job>("/training/batch/benchmark", {
                  request,
                  options,
                }),
              );
              setChosen({});
            })
          }
        >
          Run batch benchmark
        </button>
        {running && (
          <button
            type="button"
            onClick={() =>
              void action(async () => {
                if (job)
                  setJob(
                    await api<Job>(
                      `/training/batch/benchmark/${job.id}/stop`,
                      {},
                    ),
                  );
              })
            }
          >
            Cancel batch benchmark
          </button>
        )}
        {job && (
          <>
            <p>
              Benchmark {job.state};{" "}
              {appliedExperimentId === job.id
                ? "applied to next training submission; training has not started."
                : "not applied; actual training has not started."}
            </p>
            {job.error && <p role="alert">{job.error}</p>}
            <table aria-label="Batch benchmark results">
              <thead>
                <tr>
                  {[
                    "Stage",
                    "GPU microbatch",
                    "Accumulation",
                    "Effective",
                    "Examples/s",
                    "Seconds/update",
                    "Peak MiB",
                    "Skips",
                    "Cache",
                    "Status",
                  ].map((t) => (
                    <th key={t}>{t}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {job.rows.map((r, i) => (
                  <tr key={i}>
                    <td>{r.stage}</td>
                    <td>{r.microbatch}</td>
                    <td>{r.accumulation}</td>
                    <td>{r.effective_batch}</td>
                    <td>{r.examples_per_second?.toFixed(1) ?? "—"}</td>
                    <td>{r.median_seconds?.toFixed(3) ?? "—"}</td>
                    <td>
                      {r.peak_allocated
                        ? Math.round(r.peak_allocated / 2 ** 20)
                        : "not CUDA"}
                    </td>
                    <td>{r.skipped_updates ?? "—"}</td>
                    <td>{r.cache}</td>
                    <td title={r.error || r.reason}>{r.status}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            {Object.entries(job.recommendation).map(([s, r]) => (
              <p key={s}>
                {s}: recommended microbatch {r.microbatch}. {r.reason}
              </p>
            ))}
            <button
              type="button"
              disabled={
                active ||
                busy ||
                running ||
                !Object.keys(job.recommendation).length
              }
              onClick={() => void apply("recommended")}
            >
              Apply recommendation
            </button>
            {stages
              .filter(([s]) => job.rows.some((r) => r.stage === s))
              .map(([s, label]) => (
                <label key={s}>
                  {label} selected result
                  <select
                    aria-label={`${label} selected result`}
                    value={chosen[s] || ""}
                    onChange={(e) =>
                      setChosen({ ...chosen, [s]: e.target.value })
                    }
                  >
                    <option value="">Select a completed stable result</option>
                    {Array.from(
                      new Set(
                        job.rows
                          .filter(
                            (r) =>
                              r.stage === s &&
                              r.status === "completed" &&
                              !r.skipped_updates,
                          )
                          .map((r) => `${r.effective_batch}:${r.microbatch}`),
                      ),
                    ).map((v) => (
                      <option key={v} value={v}>
                        {v.split(":")[1]} GPU microbatch / effective{" "}
                        {v.split(":")[0]}
                      </option>
                    ))}
                  </select>
                </label>
              ))}
            <button
              type="button"
              disabled={
                active ||
                busy ||
                running ||
                !Object.values(chosen).some(Boolean)
              }
              onClick={() => void apply("selected")}
            >
              Apply selected result
            </button>
          </>
        )}
      </details>
      {error && <p role="alert">{error}</p>}
    </section>
  );
}
