import { LineChart } from "./ScienceCharts";
export type MeshConvergenceRecord = {
  architecture?: string;
  stage?: string;
  history?: {
    step: number;
    training_loss: number;
    optimizer_update_skipped?: boolean;
    gradients_finite?: boolean;
  }[];
  training_series?: { step: number; mean: number; count: number }[];
  training_summary?: {
    window: number;
    count: number;
    mean: number | null;
    label: string;
    family_objective: Record<
      string,
      { mean: number | null; examples: number; missing_updates: number }
    >;
    quantity_objective: string;
  };
  validation_history?: Validation[];
  latest_validation?: Validation | null;
  best_validation?: Validation | null;
  next_validation_update?: number | null;
};
type Group = {
  physical_mse: number;
  physical_rmse: number;
  normalized_mse: number;
  case_count: number;
  evaluated_points: number;
};
type Validation = {
  prediction_scope?: string;
  update?: number;
  uniform_domain_normalized_mse: number;
  decoded_field_mse?: number;
  groups?: Record<string, Group>;
};
const colors: Record<string, string> = {
  "heat/u": "#227b55",
  "wave/u": "#456cc2",
  "wave/v": "#a146a1",
};
const number = (value: number | null | undefined) =>
  value == null ? "Not available" : value.toPrecision(5);
export function MeshConvergence({ record }: { record: MeshConvergenceRecord }) {
  const rows = (record.history || []).filter(
    (r) =>
      !r.optimizer_update_skipped &&
      r.gradients_finite !== false &&
      Number.isFinite(r.training_loss),
  );
  const summary = record.training_summary,
    validation = record.validation_history || [],
    latest = record.latest_validation;
  const title = `${record.architecture || "Recorded model"} · ${record.stage || "Recorded stage"}`;
  return (
    <section aria-label="Mesh convergence">
      <p>{title}</p>
      <LineChart
        title="Optimized training objective"
        xlabel="Successful optimizer update"
        ylabel="Training objective"
        series={[
          {
            name: "Raw objective (successful updates)",
            color: "#247b75",
            opacity: 0.25,
            points: rows.map((r) => ({ x: r.step, y: r.training_loss })),
          },
          {
            name: summary?.label || "Rolling training mean unavailable",
            color: "#155a56",
            points: (record.training_series || []).map((r) => ({
              x: r.step,
              y: r.mean,
            })),
          },
        ].filter((s) => s.points.length > 0)}
      />
      {summary && (
        <>
          <p>
            {summary.label}: {number(summary.mean)} · {summary.count}/
            {summary.window} updates; partial window before enough successes.
          </p>
          <table>
            <caption>
              Family-conditioned optimized objective, same rolling window
            </caption>
            <thead>
              <tr>
                <th>Family</th>
                <th>E[L | family]</th>
                <th>Contributing examples</th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(summary.family_objective).map(([family, r]) => (
                <tr key={family}>
                  <td>{family}</td>
                  <td>
                    {number(r.mean)}
                    {r.missing_updates > 0
                      ? ` (${r.missing_updates} updates lack this breakdown)`
                      : ""}
                  </td>
                  <td>{r.examples}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <p>{summary.quantity_objective}</p>
        </>
      )}
      {validation.length > 0 ? (
        <>
          <LineChart
            title="Fixed-validation normalized decoded-field MSE"
            xlabel="Successful optimizer update"
            ylabel="Squared error / fixed TRAIN scale²"
            series={[
              {
                name: "Aggregate normalized decoded-field MSE",
                color: "#dd7d21",
                points: validation
                  .filter((r) => r.update != null)
                  .map((r) => ({
                    x: r.update!,
                    y: r.uniform_domain_normalized_mse,
                  })),
              },
              ...Object.entries(colors).map(([name, color]) => ({
                name,
                color,
                points: validation
                  .filter((r) => r.update != null && r.groups?.[name])
                  .map((r) => ({
                    x: r.update!,
                    y: r.groups![name].normalized_mse,
                  })),
              })),
            ].filter((s) => s.points.length > 0)}
          />
          <p>
            {latest?.prediction_scope ||
              "Prediction scope unavailable in this historical record."}
          </p>
          <p>
            Full recorded stage history. Markers are actual evaluations;
            connecting lines only aid reading. Fixed TRAIN scales, not per-case
            variance. Initial-only for M1; otherwise equal initial and fixed
            target-time error per case.
          </p>
          <p>
            Latest validation update: {latest?.update ?? "Unknown"}; normalized
            aggregate {number(latest?.uniform_domain_normalized_mse)}; physical
            aggregate {number(latest?.decoded_field_mse)} (mixed squared
            quantity units).
          </p>
          <p>
            Best recorded normalized aggregate:{" "}
            {number(record.best_validation?.uniform_domain_normalized_mse)} at
            update {record.best_validation?.update ?? "Unknown"}. Diagnostic
            only; no checkpoint-selection change.
          </p>
          <table>
            <caption>
              Latest fixed validation · physical and normalized errors
            </caption>
            <thead>
              <tr>
                <th>Quantity</th>
                <th>Physical MSE</th>
                <th>Physical RMSE</th>
                <th>Normalized MSE</th>
                <th>Cases / points</th>
              </tr>
            </thead>
            <tbody>
              {Object.keys(colors).map((name) => {
                const r = latest?.groups?.[name];
                return (
                  <tr key={name}>
                    <td>
                      {name === "wave/v"
                        ? "Wave v · velocity"
                        : name === "wave/u"
                          ? "Wave u · displacement"
                          : "Heat u"}
                    </td>
                    <td>{number(r?.physical_mse)}</td>
                    <td>{number(r?.physical_rmse)}</td>
                    <td>{number(r?.normalized_mse)}</td>
                    <td>
                      {r
                        ? `${r.case_count} / ${r.evaluated_points}`
                        : "Not available"}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
          <p>
            Physical u and v errors have different units/scales; use
            quantity-specific normalized errors for comparison. Older records
            may lack breakdowns.
          </p>
        </>
      ) : (
        <p>
          No validation has been recorded. Next scheduled validation update:{" "}
          {record.next_validation_update ??
            "Not available / no remaining update"}
          .
        </p>
      )}
    </section>
  );
}
