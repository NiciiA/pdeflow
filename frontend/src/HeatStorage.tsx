import { useEffect, useState } from "react";
import { call } from "./heat";
type Group = {
  id: string;
  name: string;
  bytes: number;
  active: boolean;
  selected: boolean;
  files: string[];
  model_ids: string[];
};
type Inventory = {
  total_bytes: number;
  categories: Record<string, number>;
  models: Group[];
};
type Plan = {
  id: string;
  digest: string;
  confirmation: string;
  unique_bytes: number;
  expected_last_link_allocated_bytes: number;
  models: Group[];
  entries: { path: string; bytes: number }[];
  shared_retained: { id: string; reason: string }[];
  selected_becoming_unavailable: string[];
  data_included: boolean;
  results_included: boolean;
};
const size = (n: number) =>
  n >= 1024 ** 3
    ? `${(n / 1024 ** 3).toFixed(2)} GB`
    : n >= 1024 ** 2
      ? `${(n / 1024 ** 2).toFixed(2)} MB`
      : `${(n / 1024).toFixed(1)} KB`;
export function HeatStorage({
  onDeleted,
  currentModel,
}: {
  onDeleted: (ids: string[]) => void;
  currentModel: string;
}) {
  const [inventory, setInventory] = useState<Inventory | null>(null),
    [selected, setSelected] = useState<string[]>([]),
    [data, setData] = useState(false),
    [results, setResults] = useState(false),
    [plan, setPlan] = useState<Plan | null>(null),
    [confirmation, setConfirmation] = useState(""),
    [error, setError] = useState(""),
    [busy, setBusy] = useState(false),
    [message, setMessage] = useState("");
  const [failures, setFailures] = useState<{ path: string; error: string }[]>(
    [],
  );
  useEffect(() => {
    let live = true;
    void call<Inventory>("/playground/storage")
      .then((v) => {
        if (live) setInventory(v);
      })
      .catch((e) => {
        if (live) setError(e.message);
      });
    return () => {
      live = false;
    };
  }, []);
  async function preview(all_models: boolean) {
    setBusy(true);
    setError("");
    setMessage("");
    try {
      setPlan(
        await call<Plan>("/playground/storage/preview", {
          selected,
          all_models,
          data,
          results,
        }),
      );
      setConfirmation("");
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }
  async function remove() {
    if (!plan) return;
    setBusy(true);
    setError("");
    try {
      const r = await call<{
        inventory: Inventory;
        actual_unique_bytes_removed: number;
        last_link_allocated_bytes_released: number;
        failures: { path: string; error: string }[];
        removed_model_ids: string[];
      }>("/playground/storage/delete", {
        identifier: plan.id,
        confirmation,
        review_digest: plan.digest,
      });
      setInventory(r.inventory);
      setPlan(null);
      setSelected([]);
      setMessage(
        `Removed ${size(r.actual_unique_bytes_removed)} of unique managed files; ${size(r.last_link_allocated_bytes_released)} of last-link disk blocks released. ${r.failures.length} failures. Filesystem snapshots may affect free space.`,
      );
      setFailures(r.failures);
      onDeleted(r.removed_model_ids);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }
  return (
    <section className="heat-storage">
      <h1>Storage</h1>
      <p>
        Remove model weights permanently. Your source, settings and imported
        data stay protected.
      </p>
      {error && <p role="alert">{error}</p>}
      {message && <p role="status">{message}</p>}
      {failures.length > 0 && (
        <details open>
          <summary>Files that could not be removed</summary>
          {failures.map((f) => (
            <p key={f.path}>
              {f.path}: {f.error}
            </p>
          ))}
        </details>
      )}
      {!inventory ? (
        <p role="status">Measuring managed storage and checking model files…</p>
      ) : (
        <>
          <div className="storage-totals">
            <strong>
              {size(inventory.total_bytes)}
              <small>Total managed</small>
            </strong>
            {Object.entries(inventory.categories).map(([k, v]) => (
              <span key={k}>
                {size(v)}
                <small>{k.replaceAll("_", " ")}</small>
              </span>
            ))}
          </div>
          {!plan ? (
            <>
              <div className="storage-scopes">
                <strong>Deletion scope</strong>
                <p>
                  Model weights include inference exports, optimizer/resume
                  checkpoints and training-stage weights. Shared stages are
                  retained when another model needs them.
                </p>
                <label>
                  <input
                    type="checkbox"
                    checked={data}
                    onChange={(e) => setData(e.target.checked)}
                  />
                  Also associated generated data / caches
                </label>
                <label>
                  <input
                    type="checkbox"
                    checked={results}
                    onChange={(e) => setResults(e.target.checked)}
                  />
                  Also associated experiment results / logs
                </label>
                <p>
                  Unrecognized files and externally linked imports cannot be
                  removed here.
                </p>
              </div>
              <div className="storage-list">
                {inventory.models.map((g) => (
                  <label key={g.id}>
                    <input
                      type="checkbox"
                      checked={selected.includes(g.id)}
                      disabled={g.active}
                      onChange={(e) =>
                        setSelected((s) =>
                          e.target.checked
                            ? [...s, g.id]
                            : s.filter((x) => x !== g.id),
                        )
                      }
                    />
                    <span>
                      {g.name}
                      <small>
                        {g.active
                          ? "Active · protected"
                          : g.selected || g.model_ids.includes(currentModel)
                            ? "Selected for reconstruction"
                            : "Saved model artifacts"}
                      </small>
                    </span>
                    <b>{size(g.bytes)}</b>
                  </label>
                ))}
              </div>
              {!inventory.models.length && (
                <p>
                  No managed model weights. The heat simulation remains
                  available.
                </p>
              )}
              <div className="heat-actions">
                <button
                  disabled={busy || !selected.length}
                  onClick={() => void preview(false)}
                >
                  Preview selected deletion
                </button>
                <button
                  className="danger"
                  disabled={busy || !inventory.models.length}
                  onClick={() => void preview(true)}
                >
                  Delete all model artifacts…
                </button>
              </div>
            </>
          ) : (
            <div className="deletion-preview">
              <h2>Review permanent deletion</h2>
              <p>
                <strong>{size(plan.unique_bytes)}</strong> unique logical bytes
                in {plan.entries.length} files. No trash copy will be kept.
              </p>
              <p>
                Expected last-link disk blocks released:{" "}
                {size(plan.expected_last_link_allocated_bytes)}. Links retained
                elsewhere release no blocks.
              </p>
              {plan.models.map((g) => (
                <p key={g.id}>
                  {g.name} · {g.files.length} weights/checkpoints
                </p>
              ))}
              {plan.shared_retained.map((g) => (
                <p key={g.id}>
                  {g.reason}: {g.id}
                </p>
              ))}
              <p>
                Generated data{" "}
                {plan.data_included
                  ? "included where associated and recognized"
                  : "remains"}
                . Results/logs{" "}
                {plan.results_included
                  ? "included where associated and recognized"
                  : "remain"}
                .
              </p>
              <p>
                Selected models becoming unavailable:{" "}
                {[
                  ...new Set([
                    ...plan.selected_becoming_unavailable,
                    ...plan.models
                      .filter((g) => g.model_ids.includes(currentModel))
                      .map((g) => g.name),
                  ]),
                ].join(", ") || "none"}
                .
              </p>
              <details>
                <summary>Exact artifacts</summary>
                <ul>
                  {plan.entries.map((e) => (
                    <li key={e.path}>
                      {e.path} · {size(e.bytes)}
                    </li>
                  ))}
                </ul>
              </details>
              <label>
                Type <strong>{plan.confirmation}</strong>
                <input
                  aria-label="Deletion confirmation"
                  value={confirmation}
                  onChange={(e) => setConfirmation(e.target.value)}
                />
              </label>
              <div className="heat-actions">
                <button
                  onClick={() => {
                    setPlan(null);
                    setConfirmation("");
                  }}
                >
                  Cancel deletion
                </button>
                <button
                  className="danger"
                  disabled={busy || confirmation !== plan.confirmation}
                  onClick={() => void remove()}
                >
                  Permanently delete
                </button>
              </div>
            </div>
          )}
        </>
      )}
    </section>
  );
}
