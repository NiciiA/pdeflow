import { useEffect, useState } from "react";
import { api } from "./types";
import type { Problem } from "./types";
import { Button } from "./components/ui/button";
import {
  Card,
  CardHeader,
  CardTitle,
  CardContent,
  CardDescription,
} from "./components/ui/card";
import { LineChart } from "./ScienceCharts";
type Target = {
  id: string;
  specialist: string;
  grid: Problem["grid"];
  architecture: { parameters: number };
  state: string;
  maximum_updates: number;
  updates: number;
  evaluation_available: boolean;
  compatible_discoverable_checkpoint: boolean;
  command: string;
  resolved_config: unknown;
};
type StorageEntry = {
  path: string;
  bytes: number;
  type: string;
  active: boolean;
  pinned: boolean;
  referenced: boolean;
  protected: boolean;
  run_id: string | null;
  suite_id?: string | null;
  created_at_unix?: number | null;
  updated_at_unix?: number | null;
};
type Inventory = {
  root: string;
  scanned_at: string;
  files: number;
  total_bytes: number;
  bytes_by_type: Record<string, number>;
  entries: StorageEntry[];
};
type LiveRun = {
  id: string;
  run_id: string;
  name: string | null;
  specialist: string;
  grid: Problem["grid"];
  state: string;
  data_mode: string;
  active: boolean;
  device: string;
  dtype: string;
  progress: number;
  budget: number;
  architecture: { parameters: number } | null;
  live?: {
    step?: number;
    latest?: { training_loss: number };
    unique_physical_problems?: number;
    updates_per_second?: number | null;
    generated_reference_problems_per_second?: number | null;
    eta_seconds?: number | null;
  };
  provider_summary?: { unique_training_problems_consumed: number };
  history: {
    step: number;
    training_loss?: number;
    data_wait_seconds?: number;
    neural_host_seconds?: number;
  }[];
  checkpoint_updated_at_unix: number | null;
  resume_command: string;
  evaluate_command: string;
  resolved_config: unknown;
  log_tail: string;
};
const bytes = (n: number) => `${(n / 1024 ** 2).toFixed(2)} MiB`;
export function TrainingOverview({ problem }: { problem?: Problem }) {
  const [revision, setRevision] = useState(0);
  const [targets, setTargets] = useState<Target[]>([]),
    [runs, setRuns] = useState<LiveRun[]>([]),
    [inventory, setInventory] = useState<Inventory | null>(null),
    [error, setError] = useState(""),
    [plan, setPlan] = useState<{
      entries: {
        preset: string;
        grid: number[];
        updates: number;
        device: string;
      }[];
      catalogue_version: string;
    } | null>(null),
    [cleanup, setCleanup] = useState<{
      id: string;
      review_digest: string;
      root?: string;
      entries: StorageEntry[];
      total_bytes: number;
    } | null>(null),
    [query, setQuery] = useState(""),
    [selected, setSelected] = useState("");
  useEffect(() => {
    let live = true;
    const update = () => {
      void Promise.all([
        api<{ targets: Target[] }>("/training-coverage"),
        api<LiveRun[]>("/training-runs"),
      ])
        .then(([c, r]) => {
          if (live) {
            setTargets(c.targets);
            setRuns(r);
            setError("");
          }
        })
        .catch((e) => {
          if (live) setError(String(e));
        });
    };
    update();
    const id = setInterval(update, 5000);
    return () => {
      live = false;
      clearInterval(id);
    };
  }, [revision]);
  async function storage(refresh = false) {
    try {
      setInventory(await api<Inventory>(`/storage?refresh=${refresh}`));
      setError("");
    } catch (e) {
      setError(String(e));
    }
  }
  const selectedRun =
    runs.find((r) => r.id === selected) ??
    runs.find((r) => r.active) ??
    runs[0];
  const quote = (s: string) => "'" + s.replaceAll("'", "'\"'\"'") + "'";
  const runPath =
    selectedRun?.resume_command.replace(/^pdeflow resume /, "") ?? "";
  const provider = (
    selectedRun?.resolved_config as { provider?: unknown } | undefined
  )?.provider;
  const evalPath =
    runPath + (provider ? "/evaluation/validation/0/" : "/data/");
  const prefix = provider ? "" : "test_";
  const reconstructCommand = `pdeflow reconstruct --config ${quote(runPath + "/config.json")} --checkpoint ${quote(runPath + "/training/inference.pt")} --problem ${quote(evalPath + prefix + "problem.json")} --observations ${quote(evalPath + prefix + "observations.json")} --mode observation_pde --output ${quote(runPath + "/manual-reconstruction")}`;
  const liveIsFresh =
    !!selectedRun?.active &&
    (selectedRun.live?.step ?? 0) > selectedRun.progress;
  const uniqueCounts = [
    selectedRun?.live?.unique_physical_problems,
    selectedRun?.provider_summary?.unique_training_problems_consumed,
  ].filter((v): v is number => typeof v === "number");
  const currentUnique = uniqueCounts.length
    ? Math.max(...uniqueCounts)
    : "not recorded";
  const lastHistory = selectedRun?.history
    .filter((h) => h.training_loss != null)
    .at(-1);
  const recordedLoss =
    lastHistory && lastHistory.step > (selectedRun?.live?.step ?? -1)
      ? lastHistory.training_loss
      : (selectedRun?.live?.latest?.training_loss ??
        lastHistory?.training_loss);
  const matches = (t: Target) =>
    problem?.family === t.specialist &&
    ["shape", "times", "domain", "convention", "temporal_kind"].every(
      (k) =>
        JSON.stringify(t.grid[k as keyof typeof t.grid]) ===
        JSON.stringify(problem.grid[k as keyof typeof problem.grid]),
    );
  return (
    <div className="stack">
      <Card>
        <CardHeader>
          <CardTitle>Training coverage and live overview</CardTitle>
          <CardDescription>
            Finite catalogue, direct-field conditional flow matching.
            Unsupported layouts and generative methods have no target.
            Completion is not scientific accuracy. CLI workers continue when
            this page closes.
          </CardDescription>
        </CardHeader>
        <CardContent>
          {error && <p role="alert">{error}</p>}
          <Button variant="outline" onClick={() => setRevision((v) => v + 1)}>
            Refresh training coverage & progress
          </Button>
          <Button
            variant="outline"
            onClick={() =>
              void api<NonNullable<typeof plan>>("/training-plan")
                .then(setPlan)
                .catch((e) => setError(String(e)))
            }
          >
            Plan full catalogue (no training)
          </Button>
          <code>pdeflow train-all --plan</code> ·{" "}
          <code>pdeflow train-all --missing --plan</code>
          {plan && (
            <div aria-label="Full catalogue plan">
              <p>
                {plan.catalogue_version} · {plan.entries.length} sequential
                targets · no jobs launched
              </p>
              <pre>
                {plan.entries
                  .map(
                    (e) =>
                      `${e.preset}: ${e.grid.join("×")} / ${e.updates} updates / ${e.device}`,
                  )
                  .join("\n")}
              </pre>
            </div>
          )}
          <div className="bounded-table">
            <table aria-label="Training coverage">
              <thead>
                <tr>
                  <th>Target / current request</th>
                  <th>Native layout</th>
                  <th>Budget / actual</th>
                  <th>Coverage</th>
                  <th>Command</th>
                </tr>
              </thead>
              <tbody>
                {targets.map((t) => (
                  <tr key={t.id}>
                    <td>
                      {t.id}
                      {matches(t) && <strong> · matches current layout</strong>}
                      <br />
                      {t.architecture.parameters.toLocaleString()} parameters
                    </td>
                    <td>
                      {t.grid.shape.join("×")} ·{" "}
                      {t.grid.temporal_kind === "static"
                        ? "static"
                        : `τ=[${t.grid.times.join(", ")}]`}
                      <br />
                      {t.grid.domain.boundary}; lengths{" "}
                      {t.grid.domain.lengths.join(", ")}
                    </td>
                    <td>
                      {t.updates} required / {t.maximum_updates} recorded
                    </td>
                    <td>
                      Configured · {t.state.replaceAll("_", " ")}
                      <br />
                      managed inverse evaluation{" "}
                      {t.evaluation_available ? "available" : "unavailable"}
                      <br />
                      compatible export{" "}
                      {t.compatible_discoverable_checkpoint
                        ? "discoverable"
                        : "unavailable"}
                    </td>
                    <td>
                      <code>{t.command}</code>
                      <details>
                        <summary>Resolved target</summary>
                        <pre>{JSON.stringify(t.resolved_config, null, 2)}</pre>
                      </details>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </CardContent>
      </Card>
      <Card>
        <CardHeader>
          <CardTitle>Actual managed progress</CardTitle>
          <CardDescription>
            Refreshes every five seconds. Kernel run ownership determines active
            status; a stale manifest flag does not.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <p>
            {runs.filter((r) => r.active).length} active worker(s). ETA appears
            only after at least 20 measured updates and 10 seconds.
          </p>
          <select
            aria-label="Live training run"
            value={selectedRun?.id ?? ""}
            onChange={(e) => setSelected(e.target.value)}
          >
            {runs.map((r) => (
              <option key={r.id} value={r.id}>
                {r.name || r.run_id} · {r.state}
              </option>
            ))}
          </select>
          {selectedRun && (
            <>
              <p>
                {selectedRun.run_id} · {selectedRun.specialist} ·{" "}
                {selectedRun.grid.shape.join("×")} · {selectedRun.device}/
                {selectedRun.dtype} · data mode {selectedRun.data_mode} ·{" "}
                {selectedRun.architecture?.parameters.toLocaleString() ??
                  "legacy metadata"}{" "}
                parameters
              </p>
              <p>
                {selectedRun.state} · update{" "}
                {selectedRun.active
                  ? Math.max(selectedRun.live?.step ?? 0, selectedRun.progress)
                  : selectedRun.progress}
                /{selectedRun.budget} · unique physical problems {currentUnique}{" "}
                ·{" "}
                {liveIsFresh &&
                typeof selectedRun.live?.updates_per_second === "number" &&
                (selectedRun.live.step ?? 0) > selectedRun.progress
                  ? `${selectedRun.live.updates_per_second.toFixed(2)} updates/s`
                  : "throughput unavailable until fresh session measurements"}{" "}
                · generated references/s{" "}
                {liveIsFresh &&
                selectedRun.live?.generated_reference_problems_per_second !=
                  null
                  ? selectedRun.live.generated_reference_problems_per_second.toFixed(
                      2,
                    )
                  : "unavailable"}{" "}
                · ETA{" "}
                {liveIsFresh && selectedRun.live?.eta_seconds != null
                  ? `${Math.round(selectedRun.live.eta_seconds)} s`
                  : "unavailable"}
              </p>
              <p>
                Last recorded training CFM: {recordedLoss ?? "not recorded"}.
                Fixed validation CFM is shown at its actual saved evaluation
                updates below.
              </p>
              <p>
                Latest resumable checkpoint:{" "}
                {selectedRun.checkpoint_updated_at_unix
                  ? new Date(
                      selectedRun.checkpoint_updated_at_unix * 1000,
                    ).toLocaleString()
                  : "unavailable"}
              </p>
              <code>{selectedRun.resume_command}</code>
              <br />
              <code>{selectedRun.evaluate_command}</code>
              <details>
                <summary>Copy run commands</summary>
                {[
                  selectedRun.resume_command,
                  selectedRun.evaluate_command,
                  reconstructCommand,
                ].map((command) => (
                  <div key={command}>
                    <code>{command}</code>
                    <Button
                      variant="outline"
                      onClick={() =>
                        void navigator.clipboard
                          .writeText(command)
                          .catch((e) => setError(String(e)))
                      }
                    >
                      Copy command
                    </Button>
                  </div>
                ))}
              </details>
              <LineChart
                title="Recorded data wait and neural host time"
                xlabel="Optimizer update"
                ylabel="Cumulative host seconds"
                series={[
                  {
                    name: "Data wait",
                    color: "#16786e",
                    points: selectedRun.history
                      .filter((h) => h.data_wait_seconds !== undefined)
                      .map((h) => ({ x: h.step, y: h.data_wait_seconds! })),
                  },
                  {
                    name: "Neural host time",
                    color: "#aa6b38",
                    points: selectedRun.history
                      .filter((h) => h.neural_host_seconds !== undefined)
                      .map((h) => ({ x: h.step, y: h.neural_host_seconds! })),
                  },
                ]}
              />
              <p>
                Timing includes host work; asynchronous accelerator kernels are
                not a peak-memory or isolated kernel timing measurement. CFM
                training/validation curves are below.
              </p>
              <details>
                <summary>Run configuration and relevant logs</summary>
                <pre>
                  {JSON.stringify(selectedRun.resolved_config, null, 2)}
                </pre>
                <pre>{selectedRun.log_tail}</pre>
              </details>
            </>
          )}
        </CardContent>
      </Card>
      <Card>
        <CardHeader>
          <CardTitle>Output storage</CardTitle>
          <CardDescription>
            Cached stat inventory. Refresh explicitly to rescan. Scientific
            results, exports, provenance, QA and unknown files are protected.
            Cleanup only proposes recognized disposable caches.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <Button variant="outline" onClick={() => void storage()}>
            Show cached storage
          </Button>{" "}
          <Button variant="outline" onClick={() => void storage(true)}>
            Refresh storage inventory
          </Button>{" "}
          <Button
            variant="outline"
            onClick={() =>
              void api<NonNullable<typeof cleanup>>("/storage/preview", {})
                .then(setCleanup)
                .catch((e) => setError(String(e)))
            }
          >
            Preview cleanup (no deletion)
          </Button>
          {inventory && (
            <>
              <p>
                {inventory.root} · {inventory.files} files ·{" "}
                {bytes(inventory.total_bytes)} · scanned {inventory.scanned_at}
              </p>
              <pre>
                {Object.entries(inventory.bytes_by_type)
                  .map(([k, v]) => `${k}: ${bytes(v)}`)
                  .join("\n")}
              </pre>
              <input
                aria-label="Filter storage paths"
                placeholder="Filter artifact paths"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
              />
              <div className="bounded-table">
                <table aria-label="Storage inventory">
                  <thead>
                    <tr>
                      <th>Path / related run and suite</th>
                      <th>Created / updated</th>
                      <th>Type / size</th>
                      <th>Protection</th>
                      <th>Pin</th>
                    </tr>
                  </thead>
                  <tbody>
                    {inventory.entries
                      .filter((e) => e.path.includes(query))
                      .slice(0, 100)
                      .map((e) => (
                        <tr key={e.path}>
                          <td>
                            {e.path}
                            <br />
                            {e.run_id}
                            <br />
                            {e.suite_id}
                          </td>
                          <td>
                            {e.created_at_unix != null
                              ? new Date(
                                  e.created_at_unix * 1000,
                                ).toLocaleString()
                              : "creation unavailable"}
                            <br />
                            {e.updated_at_unix != null
                              ? new Date(
                                  e.updated_at_unix * 1000,
                                ).toLocaleString()
                              : "update unavailable"}
                          </td>
                          <td>
                            {e.type} / {bytes(e.bytes)}
                          </td>
                          <td>
                            {e.active ? "active; " : ""}
                            {e.referenced ? "referenced; " : ""}
                            {e.protected ? "protected" : "cache eligible"}
                          </td>
                          <td>
                            <Button
                              variant="outline"
                              onClick={() =>
                                void api("/storage/pin", {
                                  path: e.path,
                                  pinned: !e.pinned,
                                })
                                  .then(() => storage(true))
                                  .catch((ex) => setError(String(ex)))
                              }
                            >
                              {e.pinned ? "Unpin" : "Pin"}
                            </Button>
                          </td>
                        </tr>
                      ))}
                  </tbody>
                </table>
              </div>
              <p>
                First 100 matching files shown; CLI includes the full inventory.
              </p>
            </>
          )}
          {cleanup && (
            <div aria-label="Cleanup preview">
              <p>
                Preview only: {cleanup.entries.length} cache files /{" "}
                {bytes(cleanup.total_bytes)}. Nothing deleted or relocated.
              </p>
              <pre>
                {cleanup.entries
                  .map((e) => `${e.path} / ${e.bytes} bytes`)
                  .join("\n") || "No eligible disposable caches."}
              </pre>
              <p>
                After reviewing every affected path, the CLI can move this
                specific plan into recoverable quarantine:
              </p>
              <code>
                pdeflow cleanup{" "}
                {cleanup.root ? `--root ${quote(cleanup.root)} ` : ""}--apply{" "}
                {cleanup.id} --confirm {cleanup.review_digest}
              </code>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
