import { useEffect, useState } from "react";
import { api } from "./types";
import {
  Card,
  CardHeader,
  CardTitle,
  CardContent,
  CardDescription,
} from "./components/ui/card";
export type Series = {
  name: string;
  color: string;
  points: { x: number; y: number }[];
  dashed?: boolean;
  opacity?: number;
};
export function LineChart({
  title,
  xlabel,
  ylabel,
  series,
}: {
  title: string;
  xlabel: string;
  ylabel: string;
  series: Series[];
}) {
  const all = series
    .flatMap((s) => s.points)
    .filter((p) => Number.isFinite(p.x) && Number.isFinite(p.y));
  if (!all.length)
    return (
      <div className="chart-empty">
        <h3>{title}</h3>
        <p>No recorded values are available. No curve is inferred.</p>
      </div>
    );
  const xmin = Math.min(...all.map((p) => p.x)),
    xmax = Math.max(...all.map((p) => p.x)),
    ymin = Math.min(...all.map((p) => p.y)),
    ymax = Math.max(...all.map((p) => p.y));
  const px = (x: number) => 58 + (340 * (x - xmin)) / (xmax - xmin || 1),
    py = (y: number) => 220 - (185 * (y - ymin)) / (ymax - ymin || 1);
  return (
    <figure className="science-chart">
      <h3>{title}</h3>
      <svg
        viewBox="0 0 440 285"
        role="img"
        aria-label={`${title}; horizontal axis ${xlabel}, vertical axis ${ylabel}`}
      >
        <path d="M58,30V220H400" fill="none" stroke="#8a9aaf" />
        {series.map((s) => (
          <g key={s.name}>
            <polyline
              points={s.points.map((p) => `${px(p.x)},${py(p.y)}`).join(" ")}
              fill="none"
              stroke={s.color}
              strokeWidth="2"
              strokeOpacity={s.opacity}
              strokeDasharray={s.dashed ? "5 4" : undefined}
            />
            {s.points.map((p, i) => (
              <circle
                key={i}
                cx={px(p.x)}
                cy={py(p.y)}
                r="2.4"
                fill={s.color}
                opacity={s.opacity}
              >
                <title>{`${s.name}: ${xlabel}=${p.x}, value=${p.y}`}</title>
              </circle>
            ))}
          </g>
        ))}
        <text x="58" y="239">
          {xmin.toPrecision(3)}
        </text>
        <text x="370" y="239">
          {xmax.toPrecision(3)}
        </text>
        <text x="50" textAnchor="end" y="40">
          {ymax.toPrecision(3)}
        </text>
        <text x="50" textAnchor="end" y="220">
          {ymin.toPrecision(3)}
        </text>
        <text x="165" y="270">
          {xlabel}
        </text>
        <text transform="translate(12,200) rotate(-90)">{ylabel}</text>
      </svg>
      <figcaption>
        {series.map((s) => (
          <span key={s.name}>
            <i style={{ background: s.color }} />
            {s.name}
            {s.dashed ? " (dashed)" : ""}
          </span>
        ))}
      </figcaption>
    </figure>
  );
}
export function ObservationCharts({
  points,
}: {
  points: {
    index: number;
    field: string;
    observed: number;
    predicted: number;
    residual: number;
  }[];
}) {
  if (!points.length)
    return <p>No observed values available for diagnostic plots.</p>;
  return (
    <div className="diagnostic-chart-grid">
      {[...new Set(points.map((p) => p.field))].map((field) => {
        const rows = points.filter((p) => p.field === field),
          all = rows.flatMap((p) => [p.observed, p.predicted]),
          lo = Math.min(...all),
          hi = Math.max(...all);
        const pos = (v: number) => 55 + (320 * (v - lo)) / (hi - lo || 1);
        return (
          <div key={field}>
            <figure className="science-chart">
              <h3>{field}: predicted versus observed</h3>
              <svg
                viewBox="0 0 430 280"
                role="img"
                aria-label={`${field} predicted versus observed with identity line`}
              >
                <path d="M55,30V225H380" fill="none" stroke="#8a9aaf" />
                <line
                  x1="55"
                  y1="225"
                  x2="375"
                  y2="35"
                  stroke="#8a9aaf"
                  strokeDasharray="5 4"
                />
                {rows.map((p) => (
                  <circle
                    key={p.index}
                    cx={pos(p.observed)}
                    cy={225 - (190 * (p.predicted - lo)) / (hi - lo || 1)}
                    r="4"
                    fill="#c57439"
                  >
                    <title>{`Observation${p.index + 1}: measured${p.observed}, predicted${p.predicted}`}</title>
                  </circle>
                ))}
                <text x="55" y="244">
                  {lo.toPrecision(3)}
                </text>
                <text x="350" y="244">
                  {hi.toPrecision(3)}
                </text>
                <text x="145" y="267">
                  Observed [{field} units]
                </text>
                <text transform="translate(15,198) rotate(-90)">
                  Predicted [{field} units]
                </text>
              </svg>
              <figcaption>
                Dots: actual observations and population-mean predictions.
                Dashed line: perfect agreement.
              </figcaption>
            </figure>
            <LineChart
              title={`${field}: observation residual`}
              xlabel="Observation row"
              ylabel="Prediction − observed"
              series={[
                {
                  name: "Actual residual",
                  color: "#b75b37",
                  points: rows.map((p) => ({ x: p.index + 1, y: p.residual })),
                },
              ]}
            />
          </div>
        );
      })}
    </div>
  );
}
type PlotPayload = {
  points: Parameters<typeof ObservationCharts>[0]["points"];
  traces: {
    position: number | null;
    label: string;
    values: Record<string, number>;
  }[];
  weighted: boolean;
  total_observations: number;
  downsampled: boolean;
  cost: Record<string, number | null>;
};
export function ResultCharts({ jobId }: { jobId: string }) {
  const [data, setData] = useState<PlotPayload | null>(null),
    [error, setError] = useState("");
  useEffect(() => {
    let live = true;
    void api<PlotPayload>(`/jobs/${jobId}/plots`)
      .then((v) => {
        if (live) {
          setData(v);
          setError("");
        }
      })
      .catch((e) => {
        if (live) setError(String(e));
      });
    return () => {
      live = false;
    };
  }, [jobId]);
  const names = data
    ? [...new Set(data.traces.flatMap((t) => Object.keys(t.values)))]
    : [];
  return (
    <Card>
      <CardHeader>
        <CardTitle>Observation & equation consistency</CardTitle>
        <CardDescription>
          Lower residuals mean closer agreement with measurements or the
          discretized equation. They do not prove the missing field is correct.
        </CardDescription>
      </CardHeader>
      <CardContent>
        {error && <p role="alert">{error}</p>}
        {data ? (
          <>
            <p>
              {data.weighted ? "Weighted" : "Equal-weight"} population mean ·{" "}
              {data.total_observations} measured values.{" "}
              {data.downsampled
                ? "Plot shows a bounded subset; all values remain in numerical results."
                : ""}
            </p>
            <p>
              These plots use population-mean predictions. The reported
              population RMSE averages squared residuals across candidates and
              measurements with saved weights; it also reflects candidate
              variability.
            </p>
            <ObservationCharts points={data.points} />
            {names.length > 0 && (
              <details>
                <summary>Diagnostics over generation</summary>
                {names.map((name, i) => (
                  <LineChart
                    key={name}
                    title={name}
                    xlabel="Generation coordinate s (0 → 1)"
                    ylabel="Recorded physical diagnostic"
                    series={[
                      {
                        name: "Estimated endpoint",
                        color: ["#4275ad", "#a36938"][i % 2],
                        points: data.traces
                          .filter(
                            (t) =>
                              t.label.includes("estimated") &&
                              t.position !== null &&
                              name in t.values,
                          )
                          .map((t) => ({ x: t.position!, y: t.values[name] })),
                      },
                      {
                        name: "Returned fields",
                        color: "#336d55",
                        points: data.traces
                          .filter(
                            (t) =>
                              !t.label.includes("estimated") &&
                              t.position !== null &&
                              name in t.values,
                          )
                          .map((t) => ({ x: t.position!, y: t.values[name] })),
                      },
                    ]}
                  />
                ))}
              </details>
            )}
            <details>
              <summary>Measured computational cost</summary>
              <dl className="plain-metrics">
                {Object.entries(data.cost).map(([name, value]) => (
                  <div key={name}>
                    <dt>{name.replaceAll("_", " ")}</dt>
                    <dd>
                      {typeof value === "number"
                        ? value.toPrecision(4)
                        : "Unavailable"}
                    </dd>
                  </div>
                ))}
              </dl>
              <p>
                Batch calls and particle evaluations are distinct. Generation
                steps do not refine the physical grid.
              </p>
            </details>
          </>
        ) : (
          <p role="status">Loading recorded diagnostics…</p>
        )}
      </CardContent>
    </Card>
  );
}
