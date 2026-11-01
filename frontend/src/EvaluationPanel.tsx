import { useState } from "react";
import { LineChart } from "./ScienceCharts";
type Errors = {
  mse?: number | null;
  rmse?: number | null;
  mae?: number | null;
  relative_l2?: number | null;
  status?: string;
  point_count?: number;
};
type Field = {
  units: string;
  ensemble_mean: Record<string, Errors>;
  weighted_mean_of_sample_errors: Record<string, Errors>;
  per_sample: Record<string, Errors>[];
  per_time: ({ physical_time: number | null } & Record<string, unknown>)[];
  empirical_rms_spread: number;
};
type Report = {
  fields: Record<string, Field>;
  summary_rule: string;
  relative_norm_rule: string;
  comparison: Record<string, unknown>;
  observation_exclusion: string;
};
function num(v: number | null | undefined) {
  return v === null || v === undefined ? "undefined" : v.toPrecision(5);
}
export function EvaluationPanel({
  metrics,
}: {
  metrics: Record<string, unknown>;
}) {
  const [metric, setMetric] = useState<"mse" | "rmse" | "relative_l2">("rmse");
  const report = metrics.evaluation as Report | undefined;
  if (!report)
    return (
      <p>
        Attach a retained reference with Compare to evaluate physical
        reconstruction errors.
      </p>
    );
  return (
    <section className="evaluation-panel">
      <h3>Reconstruction versus reference</h3>
      <p>{report.summary_rule}</p>
      <p>{report.relative_norm_rule}</p>
      <p>{report.observation_exclusion}</p>
      <div className="bounded-table">
        <table aria-label="Reference comparison errors">
          <thead>
            <tr>
              <th>Field / units</th>
              <th>Summary / region</th>
              <th>MSE</th>
              <th>RMSE</th>
              <th>MAE</th>
              <th>Relative L2</th>
            </tr>
          </thead>
          <tbody>
            {Object.entries(report.fields).flatMap(([name, f]) => [
              ...Object.entries(f.ensemble_mean).map(([region, e]) => (
                <tr key={`${name}-${region}`}>
                  <td>
                    {name} [{f.units}]
                  </td>
                  <td>
                    Weighted mean / {region} ({e.point_count} points)
                  </td>
                  <td>{num(e.mse)}</td>
                  <td>{num(e.rmse)}</td>
                  <td>{num(e.mae)}</td>
                  <td>
                    {num(e.relative_l2)} {e.status !== "ok" ? e.status : ""}
                  </td>
                </tr>
              )),
              ...Object.entries(f.weighted_mean_of_sample_errors).map(
                ([region, e]) => (
                  <tr key={`${name}-individual-${region}`}>
                    <td>{name}</td>
                    <td>Mean individual errors / {region}</td>
                    <td>{num(e.mse)}</td>
                    <td>{num(e.rmse)}</td>
                    <td>{num(e.mae)}</td>
                    <td>{num(e.relative_l2)}</td>
                  </tr>
                ),
              ),
            ])}
          </tbody>
        </table>
      </div>
      <label>
        Per-time error metric
        <select
          aria-label="Per-time error metric"
          value={metric}
          onChange={(e) => setMetric(e.target.value as typeof metric)}
        >
          <option value="mse">MSE</option>
          <option value="rmse">RMSE</option>
          <option value="relative_l2">Relative L2</option>
        </select>
      </label>
      <LineChart
        title="Reference error by physical time"
        xlabel="Physical time τ (static uses one index)"
        ylabel={`${metric} · separate field units`}
        series={Object.entries(report.fields).map(([name, f], i) => ({
          name,
          color: ["#426eab", "#aa6b38", "#247c75", "#775994"][i % 4],
          points: f.per_time
            .filter((t) => (t.full as Errors)[metric] != null)
            .map((t, i) => ({
              x: t.physical_time ?? i,
              y: (t.full as Errors)[metric]!,
            })),
        }))}
      />
      <details>
        <summary>Individual sample metrics and empirical spread</summary>
        <pre>
          {JSON.stringify(
            Object.fromEntries(
              Object.entries(report.fields).map(([name, f]) => [
                name,
                {
                  per_sample: f.per_sample,
                  empirical_rms_spread: f.empirical_rms_spread,
                },
              ]),
            ),
            null,
            2,
          )}
        </pre>
      </details>
      <details>
        <summary>
          Comparison grids, coordinate/time transfer and solver diagnostics
        </summary>
        <pre>
          {JSON.stringify(
            {
              comparison: report.comparison,
              reference_diagnostics: metrics.reference_diagnostics,
            },
            null,
            2,
          )}
        </pre>
      </details>
    </section>
  );
}
