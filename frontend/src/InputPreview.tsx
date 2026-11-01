/* eslint-disable react-hooks/set-state-in-effect -- Synchronize the externally validated problem, selected table row, and network request status. Responses are cancelled by identity. */
import { useEffect, useState, useId } from "react";
import { api } from "./types";
import type { Problem, Row } from "./types";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
  CardDescription,
} from "./components/ui/card";
export type Preview = {
  field: string;
  choices: { name: string; role: string; units: string }[];
  shape: number[];
  axes: string[];
  coordinates: Record<string, number[]>;
  values: number[] | number[][] | null;
  points: (Row & { row_id: number })[];
  point_count: number;
  total_observations: number;
  physical_time: number | null;
  slice_coordinate: number | null;
  units: string;
  role: string;
  valid: boolean;
  issues: { row: number | null; message: string }[];
  downsampled: boolean;
  boundary: string;
  interpretation: string;
};
export function SparsePlot({
  data,
  selected,
  onSelect,
}: {
  data: Preview;
  selected: number | null;
  onSelect: (n: number) => void;
}) {
  const pattern = useId().replaceAll(":", "");
  const [x, y] = data.axes;
  const xc = data.coordinates[x],
    yc = data.coordinates[y] ?? [];
  const pos = (v: number, coords: number[], span: number) =>
    ((v - coords[0]) / ((coords.at(-1) ?? 1) - coords[0] || 1)) * span;
  const actual = (
    data.values ?? data.points.map((p) => Number(p.value))
  ).flat() as number[];
  const lo = actual.length ? Math.min(...actual) : 0,
    hi = actual.length ? Math.max(...actual) : 1;
  const py = (v: number) => 220 - (170 * (v - lo)) / (hi - lo || 1);
  const cols = data.shape[0],
    rows = data.shape[1] ?? 1;
  return (
    <figure className="input-figure">
      <svg
        viewBox="0 0 430 285"
        role="img"
        aria-label={`Input preview for ${data.field}: ${data.values ? "actual known field" : "unknown regions with measured points"}, axes ${data.axes.join(", ")}`}
      >
        <defs>
          <pattern
            id={pattern}
            patternUnits="userSpaceOnUse"
            width="8"
            height="8"
          >
            <rect width="8" height="8" fill="#f3f6f8" />
            <path d="M0,8L8,0" stroke="#e2e8ee" />
          </pattern>
        </defs>
        <rect
          x="55"
          y="25"
          width="330"
          height="200"
          fill={`url(#${pattern})`}
          stroke="#c6d0dc"
        />
        {data.values &&
          (data.shape.length === 1 ? (
            <polyline
              fill="none"
              stroke="#426eab"
              strokeWidth="2"
              points={(data.values as number[])
                .map((v, i) => `${55 + pos(xc[i], xc, 330)},${py(v)}`)
                .join(" ")}
            />
          ) : (
            Array.from({ length: cols }, (_, i) =>
              Array.from({ length: rows }, (_, j) => {
                const v = (data.values as number[][])[i][j];
                return (
                  <rect
                    key={`${i}-${j}`}
                    x={55 + (i * 330) / cols}
                    y={25 + ((rows - j - 1) * 200) / rows}
                    width={331 / cols}
                    height={201 / rows}
                    fill={`hsl(213 45% ${88 - (55 * (v - lo)) / (hi - lo || 1)}%)`}
                  />
                );
              }),
            )
          ))}
        {data.points.map((p) => {
          const cx =
            data.shape.length === 1
              ? 55 + pos(Number(p[x]), xc, 330)
              : 55 +
                165 / cols +
                pos(Number(p[x]), xc, (330 * (cols - 1)) / cols);
          const cy =
            data.shape.length === 1
              ? py(Number(p.value))
              : 225 -
                100 / rows -
                pos(Number(p[y]), yc, (200 * (rows - 1)) / rows);
          return (
            <g
              key={p.row_id}
              role="button"
              tabIndex={0}
              aria-label={`Select observation ${p.row_id + 1}: ${p.field} = ${p.value}`}
              onClick={() => onSelect(p.row_id)}
              onKeyDown={(e) => {
                if (e.key === "Enter" || e.key === " ") onSelect(p.row_id);
              }}
            >
              <circle
                cx={cx}
                cy={cy}
                r={selected === p.row_id ? 8 : 5}
                fill={selected === p.row_id ? "#a43c24" : "#d57b38"}
                stroke="white"
                strokeWidth="2"
              />
              <title>{`Row${p.row_id + 1}: ${p.value}`}</title>
            </g>
          );
        })}
        <text x="55" y="245">
          {xc[0]?.toPrecision(3)}
        </text>
        <text x="352" y="245">
          {xc.at(-1)?.toPrecision(3)}
        </text>
        <text x="185" y="270">
          {x} [coordinate]
        </text>
        <text transform="translate(18,185) rotate(-90)">
          {y ? `${y} [coordinate]` : `${data.field} [${data.units}]`}
        </text>
        {y && (
          <>
            <text x="50" textAnchor="end" y="225">
              {yc[0]?.toPrecision(2)}
            </text>
            <text x="50" textAnchor="end" y="33">
              {yc.at(-1)?.toPrecision(2)}
            </text>
          </>
        )}
      </svg>
      <figcaption>
        <span className="legend-dot observation" /> Measurements ·{" "}
        <span className="legend-dot known" /> Known input · hatched area:
        unknown.{" "}
        {data.boundary === "periodic"
          ? "Periodic: opposite edges connect."
          : ""}{" "}
        {data.values
          ? `Known values ${lo.toPrecision(3)}–${hi.toPrecision(3)}.`
          : "No interpolation or inferred field is shown."}
      </figcaption>
    </figure>
  );
}
export function InputPreview({
  problem,
  rows,
  noise,
  selected,
  onSelect,
  onIssues,
}: {
  problem: Problem | null;
  rows: Row[];
  noise: number;
  selected: number | null;
  onSelect: (n: number) => void;
  onIssues?: (v: Preview["issues"]) => void;
}) {
  const [field, setField] = useState(""),
    [time, setTime] = useState(0),
    [axis, setAxis] = useState(2),
    [index, setIndex] = useState(0),
    [data, setData] = useState<Preview | null>(null),
    [error, setError] = useState(""),
    [loading, setLoading] = useState(false);
  const key = JSON.stringify({
    problem,
    rows,
    noise,
    field,
    time,
    axis,
    index,
  });
  useEffect(() => {
    if (!problem) return;
    const names = [
      ...problem.fields.map((f) => f.name),
      ...Object.keys(problem.known_fields ?? {}),
    ];
    setField((current) => (names.includes(current) ? current : names[0]));
    setTime(0);
    setIndex(0);
  }, [problem]);
  useEffect(() => {
    if (!problem || selected === null || !rows[selected]) return;
    const row = rows[selected];
    setField(String(row.field));
    if (problem.grid.temporal_kind !== "static")
      setTime(
        Math.max(
          0,
          problem.grid.times.findIndex(
            (t) => Math.abs(t - Number(row.time)) < 1e-8,
          ),
        ),
      );
    if (problem.grid.shape.length === 3) {
      const a = ["x", "y", "z"][axis],
        h = problem.grid.domain.lengths[axis] / problem.grid.shape[axis];
      setIndex(
        Math.max(
          0,
          Math.min(
            problem.grid.shape[axis] - 1,
            Math.round(
              (Number(row[a]) - problem.grid.domain.origins[axis]) / h -
                (problem.grid.convention === "periodic_cell_centers" ? 0.5 : 0),
            ),
          ),
        ),
      );
    }
  }, [selected, rows, problem, axis]);
  useEffect(() => {
    let live = true;
    if (!problem || !field) {
      setData(null);
      return;
    }
    setLoading(true);
    const timer = setTimeout(() => {
      void api<Preview>("/input-preview", {
        problem,
        rows,
        noise_std: noise,
        field,
        time_index: time,
        axis,
        index,
      })
        .then((v) => {
          if (live) {
            setData(v);
            setError("");
            onIssues?.(v.issues);
          }
        })
        .catch((e) => {
          if (live) setError(String(e));
        })
        .finally(() => {
          if (live) setLoading(false);
        });
    }, 180);
    return () => {
      live = false;
      clearTimeout(timer);
    };
    // JSON identity prevents redundant previews when canonical props are unchanged.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key]);
  return (
    <Card className="input-preview">
      <CardHeader>
        <CardTitle>Supplied information</CardTitle>
        <CardDescription>
          Measurements constrain the reconstruction; unmeasured regions remain
          unknown.
        </CardDescription>
      </CardHeader>
      <CardContent>
        {!problem ? (
          <p>
            Enter a supported equation to preview its domain and supplied
            fields. Your draft is preserved.
          </p>
        ) : (
          <>
            <div className="preview-controls">
              <label>
                Field / input
                <select
                  aria-label="Input preview field"
                  value={field}
                  onChange={(e) => {
                    setField(e.target.value);
                    onIssues?.([]);
                  }}
                >
                  {[
                    ...problem.fields.map((f) => ({
                      name: f.name,
                      role: "unknown target",
                    })),
                    ...Object.keys(problem.known_fields ?? {}).map((name) => ({
                      name,
                      role: "known input",
                    })),
                  ].map((f) => (
                    <option key={f.name} value={f.name}>
                      {f.name} · {f.role}
                    </option>
                  ))}
                </select>
              </label>
              {problem.grid.temporal_kind !== "static" && (
                <label>
                  Physical time τ
                  <select
                    aria-label="Input physical time"
                    value={time}
                    onChange={(e) => setTime(Number(e.target.value))}
                  >
                    {problem.grid.times.map((t, i) => (
                      <option key={i} value={i}>
                        {t}
                      </option>
                    ))}
                  </select>
                </label>
              )}
              {problem.grid.shape.length === 3 && (
                <>
                  <label>
                    Slice normal
                    <select
                      aria-label="Input slice normal"
                      value={axis}
                      onChange={(e) => {
                        setAxis(Number(e.target.value));
                        setIndex(0);
                      }}
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
                      aria-label="Input slice index"
                      type="range"
                      min="0"
                      max={problem.grid.shape[axis] - 1}
                      value={index}
                      onChange={(e) => setIndex(Number(e.target.value))}
                    />
                  </label>
                </>
              )}
            </div>
            {error && <p role="alert">{error}</p>}
            {loading && <p role="status">Updating input preview…</p>}
            {data && (
              <>
                <p className="preview-context">
                  {data.role} ·{" "}
                  {data.physical_time === null
                    ? "static field"
                    : `physical τ=${data.physical_time}`}
                  {data.slice_coordinate !== null
                    ? ` · normal coordinate ${data.slice_coordinate}`
                    : ""}
                </p>
                <SparsePlot
                  data={data}
                  selected={selected}
                  onSelect={onSelect}
                />
                <p>
                  {data.point_count} of {data.total_observations} measurements
                  on this field/time/slice.{" "}
                  {data.downsampled
                    ? "Display is bounded/downsampled; original input is retained."
                    : ""}
                </p>
                {data.issues.map((e, i) => (
                  <p role="alert" key={i}>
                    {e.message}
                  </p>
                ))}
              </>
            )}
          </>
        )}
      </CardContent>
    </Card>
  );
}

export function SavedInputPreview({
  jobId,
  field,
  time,
  axis,
  index,
}: {
  jobId: string;
  field: string;
  time: number;
  axis: number;
  index: number;
}) {
  const [data, setData] = useState<Preview | null>(null),
    [error, setError] = useState("");
  useEffect(() => {
    let live = true;
    void api<Preview>(
      `/jobs/${jobId}/input-preview?field=${encodeURIComponent(field)}&time_index=${time}&axis=${axis}&index=${index}`,
    )
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
  }, [jobId, field, time, axis, index]);
  return (
    <section className="plot saved-input">
      <h3>Input observations · this saved run</h3>
      {error ? (
        <p role="alert">{error}</p>
      ) : data ? (
        <>
          <SparsePlot data={data} selected={null} onSelect={() => {}} />
          <p className="plot-scale">
            {data.point_count} measurements on the selected field/time/slice.
            Unknown regions stay hatched.
          </p>
        </>
      ) : (
        <p>Loading supplied information…</p>
      )}
    </section>
  );
}
