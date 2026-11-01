import { useState } from "react";
import { Card, CardHeader, CardTitle, CardContent } from "./components/ui/card";
import { Button } from "./components/ui/button";

type Entry = { value: number | null; units: string; status: string };
type Event = {
  index: number;
  action: string;
  reason: string;
  step?: number;
  generation_position?: number;
  budget: Record<string, number>;
  failures?: string[];
};
type Process = {
  stop_reason: string;
  criteria_met: boolean;
  final_grid: { shape: number[] };
  actions: Event[];
  consumed: Record<string, number>;
  resampling: string;
};
type Verification = {
  label: string;
  entries: Record<string, Entry>;
  mandatory_failures: string[];
};

export function ProcessControls({
  value,
  onChange,
  nativePoints,
}: {
  value: unknown;
  nativePoints?: number;
  onChange: (value: unknown) => void;
}) {
  const [draft, setDraft] = useState("");
  const [error, setError] = useState("");
  const enabled = value !== null && value !== undefined;
  return (
    <details>
      <summary>Feedback controller and budgets</summary>
      <label>
        <input
          aria-label="Enable feedback controller"
          type="checkbox"
          checked={enabled}
          onChange={(e) => {
            onChange(e.target.checked ? {} : null);
            setDraft("");
          }}
        />{" "}
        Enable bounded feedback
      </label>
      {enabled && nativePoints != null && (
        <div aria-label="Native grid controller budget">
          <p>
            Native spatial grid: {nativePoints.toLocaleString()} nodes. Current
            controller grid budget:{" "}
            {Number(
              (value as { max_grid_points?: number }).max_grid_points ?? 32768,
            ).toLocaleString()}
            . Physical-time channels and particles add memory separately.
          </p>
          {nativePoints >
            Number(
              (value as { max_grid_points?: number }).max_grid_points ?? 32768,
            ) && (
            <>
              <p role="alert">
                Review the controller budget for this native grid before
                validation. The grid is never silently reduced; this limit is
                not a total device-memory estimate.
              </p>
              <Button
                variant="outline"
                onClick={() => {
                  onChange({
                    ...(value as object),
                    max_grid_points: nativePoints,
                  });
                  setDraft("");
                }}
              >
                Use requested native grid budget ({nativePoints})
              </Button>
            </>
          )}
        </div>
      )}
      <p>
        Checks distinguish an estimated endpoint from the actual final fields.
        Without explicit component thresholds, completion reports budget
        exhausted, never criteria met. Thresholds require physical/noise scales
        appropriate to this problem.
      </p>
      <p>
        FK resampling is unavailable in gradient mode; select explicit FK mode
        for compatible checkpoints. Heat refinement requires a separate
        compatible 2× nodal checkpoint and at least two total attempts; it
        restarts from fresh Gaussian noise with the same measurements.
      </p>
      {enabled && (
        <>
          <label>
            Advanced process configuration
            <textarea
              aria-label="Process configuration JSON"
              rows={10}
              value={draft || JSON.stringify(value, null, 2)}
              onChange={(e) => setDraft(e.target.value)}
            />
          </label>
          <Button
            variant="outline"
            onClick={() => {
              try {
                const next = JSON.parse(draft || JSON.stringify(value));
                if (!next || typeof next !== "object" || Array.isArray(next))
                  throw Error("Expected an object");
                onChange(next);
                setDraft("");
                setError("");
              } catch (e) {
                setError(String(e));
              }
            }}
          >
            Apply process configuration
          </Button>
          {error && <p role="alert">{error}</p>}
          <p>
            Keys: thresholds, check_interval, max_network_evaluations,
            max_sampling_steps, max_guidance_steps, max_attempts,
            max_refinement_levels (0 or 1), max_grid_points, max_particles,
            wall_seconds, patience, stagnation_relative, refine, fine_model_id,
            request_resampling. API validates all settings. Resolved defaults
            appear in the final report.
          </p>
        </>
      )}
    </details>
  );
}

export function ProcessPanel({
  metrics,
}: {
  metrics: Record<string, unknown>;
}) {
  const process = metrics.process as Process | null;
  const verification = metrics.verification as Verification | null;
  if (!verification) return null;
  return (
    <Card className="process-panel">
      <CardHeader>
        <CardTitle>Process verification</CardTitle>
      </CardHeader>
      <CardContent>
        <p aria-label="Process completion status">
          {process?.stop_reason ??
            "Schedule completed · no criteria-based controller"}
        </p>
        <p>
          {process?.criteria_met
            ? "Configured physical and observation criteria met on final fields; reconstruction accuracy is not certified."
            : "Execution finished without a criteria-met certificate. Unconfigured or violated mandatory checks remain unresolved."}
        </p>
        {process && (
          <p>
            Final physical grid: {process.final_grid.shape.join(" × ")} ·{" "}
            {process.resampling === "unavailable; FK absent"
              ? "No resampling in this gradient run; FK is a separate method"
              : process.resampling}
          </p>
        )}
        <p>
          Evaluation: {verification.label}. Mandatory unresolved checks:{" "}
          {verification.mandatory_failures.join(", ") || "none"}.
        </p>
        <table aria-label="Component verification">
          <thead>
            <tr>
              <th>Component</th>
              <th>Value</th>
              <th>Units</th>
              <th>Status</th>
            </tr>
          </thead>
          <tbody>
            {Object.entries(verification.entries).map(([name, e]) => (
              <tr key={name}>
                <td>{name}</td>
                <td>
                  {e.value === null ? "unavailable" : e.value.toPrecision(5)}
                </td>
                <td>{e.units}</td>
                <td>{e.status}</td>
              </tr>
            ))}
          </tbody>
        </table>
        {process && (
          <>
            <p>Budget consumed: {JSON.stringify(process.consumed)}</p>
            <details open>
              <summary>Execution decisions</summary>
              <table aria-label="Execution decisions">
                <thead>
                  <tr>
                    <th>Step / generation position</th>
                    <th>Action</th>
                    <th>Reason and budget</th>
                  </tr>
                </thead>
                <tbody>
                  {process.actions.map((e) => (
                    <tr key={e.index}>
                      <td>
                        {e.step ?? "candidate boundary"} /{" "}
                        {e.generation_position ?? "final"}
                      </td>
                      <td>{e.action}</td>
                      <td>
                        {e.reason}
                        <br />
                        {JSON.stringify(e.budget)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </details>
          </>
        )}
      </CardContent>
    </Card>
  );
}
