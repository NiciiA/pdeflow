import { useEffect, useState } from "react";
import { api, saveFile } from "./types";
import type { Problem, Slice } from "./types";
import { Button } from "./components/ui/button";
import { GenerationPlayer } from "./GenerationPlayer";
type View = Slice & {
  normal: number | null;
  native_shape: number[];
  display_downsampled: boolean;
};
type Inspection = {
  statistic: string;
  point: number[];
  coordinate: Record<string, number>;
  field: string;
  physical_time: number | null;
  views: View[];
  diagnostics: {
    formula: string;
    terms: Record<string, number>;
    fields: Record<string, number>;
    meaning: string;
    flux?: unknown;
  };
  observations_at_point: {
    measured: number;
    predicted: number;
    residual: number;
  }[];
  reference_value: number | null;
  meaning: string;
};
function shade(v: number, lo: number, hi: number, error = false) {
  const t = Math.max(0, Math.min(1, (v - lo) / (hi - lo || 1)));
  return error
    ? `hsl(22 62% ${97 - t * 62}%)`
    : `hsl(${205 - t * 35} ${20 + t * 35}% ${95 - t * 55}%)`;
}
function LinkedPlot({
  view,
  values,
  title,
  point,
  onPoint,
  error = false,
  surface = false,
}: {
  view: View;
  values: Slice["values"];
  title: string;
  point: number[];
  onPoint: (p: number[]) => void;
  error?: boolean;
  surface?: boolean;
}) {
  const nx = view.shape[0],
    ny = view.shape[1] ?? 1,
    flat = values.flat(),
    lo = error ? 0 : view.vmin,
    hi = error ? Math.max(...flat) : view.vmax;
  const get = (i: number, j: number) =>
    view.shape.length === 1
      ? (values as number[])[i]
      : (values as number[][])[i][j];
  const axes = view.axes.map((a) => ["x", "y", "z"].indexOf(a));
  const strides = axes.map((a) => Math.ceil(view.native_shape[a] / 64));
  const select = (i: number, j: number) => {
    const p = [...point];
    p[axes[0]] = Math.min(view.native_shape[axes[0]] - 1, i * strides[0]);
    if (axes[1] !== undefined)
      p[axes[1]] = Math.min(view.native_shape[axes[1]] - 1, j * strides[1]);
    onPoint(p);
  };
  const ix = Math.round(point[axes[0]] / strides[0]),
    iy = axes[1] === undefined ? 0 : Math.round(point[axes[1]] / strides[1]);
  const project = (i: number, j: number) => [
    205 + (i / (nx - 1 || 1) - j / (ny - 1 || 1)) * 150,
    80 +
      (i / (nx - 1 || 1) + j / (ny - 1 || 1)) * 70 -
      (60 * (get(i, j) - lo)) / (hi - lo || 1),
  ];
  return (
    <figure className="linked-figure">
      <figcaption>
        {title}
        {surface ? " · scalar height surface (2D slice)" : ""}
      </figcaption>
      <svg
        viewBox="0 0 420 300"
        role="img"
        aria-label={`${title}, linked ${surface ? "surface" : view.axes.join(" / ")} view`}
        onPointerMove={(e) => {
          if (surface) return;
          const r = e.currentTarget.getBoundingClientRect();
          select(
            Math.max(
              0,
              Math.min(
                nx - 1,
                Math.floor(
                  ((((e.clientX - r.left) / r.width) * 420 - 42) / 340) * nx,
                ),
              ),
            ),
            Math.max(
              0,
              Math.min(
                ny - 1,
                Math.floor(
                  (1 - (((e.clientY - r.top) / r.height) * 300 - 25) / 220) *
                    ny,
                ),
              ),
            ),
          );
        }}
        onClick={(e) => {
          if (!surface) return;
          const r = e.currentTarget.getBoundingClientRect(),
            x = ((e.clientX - r.left) / r.width) * 420,
            y = ((e.clientY - r.top) / r.height) * 300;
          let best = [0, 0],
            dist = Infinity;
          for (let i = 0; i < nx; i += Math.max(1, Math.ceil(nx / 32)))
            for (let j = 0; j < ny; j += Math.max(1, Math.ceil(ny / 32))) {
              const p = project(i, j),
                d = (p[0] - x) ** 2 + (p[1] - y) ** 2;
              if (d < dist) {
                dist = d;
                best = [i, j];
              }
            }
          select(best[0], best[1]);
        }}
      >
        {surface ? (
          Array.from({ length: Math.min(nx, 32) }, (_, a) => {
            const i = Math.floor(
              (a * (nx - 1)) / Math.max(1, Math.min(nx, 32) - 1),
            );
            return (
              <g key={a}>
                <polyline
                  points={Array.from({ length: ny }, (_, j) =>
                    project(i, j).join(","),
                  ).join(" ")}
                  fill="none"
                  stroke="#baad99"
                  strokeWidth=".7"
                />
                <polyline
                  points={Array.from({ length: nx }, (_, k) =>
                    project(
                      k,
                      Math.min(
                        ny - 1,
                        Math.round(
                          (a * (ny - 1)) / Math.max(1, Math.min(nx, 32) - 1),
                        ),
                      ),
                    ).join(","),
                  ).join(" ")}
                  fill="none"
                  stroke="#baad99"
                  strokeWidth=".7"
                />
              </g>
            );
          })
        ) : view.shape.length === 1 ? (
          <polyline
            points={(values as number[])
              .map(
                (v, i) =>
                  `${42 + (i / (nx - 1)) * 340},${245 - (220 * (v - lo)) / (hi - lo || 1)}`,
              )
              .join(" ")}
            fill="none"
            stroke="#246b87"
            strokeWidth="2"
          />
        ) : (
          Array.from({ length: nx }, (_, i) =>
            Array.from({ length: ny }, (_, j) => (
              <rect
                key={`${i}-${j}`}
                x={42 + (340 * i) / nx}
                y={25 + (220 * (ny - 1 - j)) / ny}
                width={341 / nx}
                height={221 / ny}
                fill={shade(get(i, j), lo, hi, error)}
              />
            )),
          )
        )}
        {!surface &&
          view.observations.map((row, k) => {
            const x = view.axes[0],
              y = view.axes[1],
              xc = view.coordinates[x],
              yc = y ? view.coordinates[y] : [0, 1];
            const xp =
              42 +
              (340 * (Number(row[x]) - xc[0])) /
                (xc.at(-1)! - xc[0] + (xc[1] - xc[0] || 1));
            const yp = y
              ? 245 -
                (220 * (Number(row[y]) - yc[0])) /
                  (yc.at(-1)! - yc[0] + (yc[1] - yc[0] || 1))
              : 245 - (220 * (Number(row.value) - lo)) / (hi - lo || 1);
            return (
              <circle
                key={k}
                cx={xp}
                cy={yp}
                r="3"
                fill="#d27836"
                stroke="#fff"
                strokeWidth="1"
              >
                <title>Measured {String(row.value)}</title>
              </circle>
            );
          })}
        <circle
          cx={
            surface
              ? project(Math.min(nx - 1, ix), Math.min(ny - 1, iy))[0]
              : 42 + (340 * (ix + 0.5)) / nx
          }
          cy={
            surface
              ? project(Math.min(nx - 1, ix), Math.min(ny - 1, iy))[1]
              : view.shape.length === 1
                ? 245 -
                  (220 * (get(Math.min(nx - 1, ix), 0) - lo)) / (hi - lo || 1)
                : 25 + (220 * (ny - iy - 0.5)) / ny
          }
          r="5"
          fill="#d27836"
          stroke="white"
          strokeWidth="2"
        />
        <text x="42" y="275">
          {view.axes[0]} · {view.coordinates[view.axes[0]][0]?.toPrecision(3)} →{" "}
          {view.coordinates[view.axes[0]].at(-1)?.toPrecision(3)}
        </text>
        <text x="260" y="295">
          {lo.toPrecision(3)} — {hi.toPrecision(3)}
        </text>
      </svg>
      <small>
        {view.display_downsampled
          ? `Display ${view.shape.join("×")}; native ${view.native_shape.join("×")}. `
          : ""}
        {error
          ? "Separate absolute-error scale."
          : "Shared reference/reconstruction scale."}
      </small>
    </figure>
  );
}
export function ScientificWorkspace({
  jobId,
  problem,
  field,
  time,
  onTime,
  sample,
}: {
  jobId: string;
  problem: Problem;
  field: number;
  time: number;
  onTime: (t: number) => void;
  sample: number;
}) {
  const [point, setPoint] = useState(() =>
      problem.grid.shape.map((n) => Math.floor(n / 2)),
    ),
    [locked, setLocked] = useState(false),
    [data, setData] = useState<Inspection | null>(null),
    [error, setError] = useState(""),
    [surface, setSurface] = useState(true),
    [mode, setMode] = useState("Spatial inspection"),
    [playing, setPlaying] = useState(false),
    [truth, setTruth] = useState(true);
  const key = JSON.stringify(point);
  useEffect(() => {
    let live = true;
    const timer = setTimeout(() => {
      void api<Inspection>(
        `/jobs/${jobId}/inspection?field_index=${field}&time_index=${time}&point=${point.join(",")}&sample_index=${sample}`,
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
    }, 120);
    return () => {
      live = false;
      clearTimeout(timer);
    };
  }, [jobId, field, time, key, sample]); // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => {
    if (
      !playing ||
      mode !== "Physical evolution" ||
      problem.grid.temporal_kind === "static"
    )
      return;
    const timer = setInterval(
      () => onTime((time + 1) % problem.grid.times.length),
      650,
    );
    return () => clearInterval(timer);
  }, [playing, mode, time, onTime, problem]);
  const changePoint = (p: number[]) => {
    if (!locked) setPoint(p);
  };
  return (
    <section
      className="mathematical-workspace"
      aria-label="Linked mathematical workspace"
    >
      <div className="math-heading">
        <span className="eyebrow">FIELD / PROCESS / EXPLANATION</span>
        <h2>
          {data?.diagnostics.formula ?? "Inspect the reconstructed field"}
        </h2>
        <p>
          {data?.statistic}. Explore what is measured, what is inferred, and how
          the generator arrived here.
        </p>
      </div>
      <div
        className="visual-modes"
        role="group"
        aria-label="Scientific view mode"
      >
        {[
          "Spatial inspection",
          "Physical evolution",
          "Generative evolution",
        ].map((v) => (
          <Button
            key={v}
            variant={mode === v ? "default" : "ghost"}
            onClick={() => {
              setMode(v);
              setPlaying(false);
            }}
          >
            {v}
          </Button>
        ))}
      </div>
      {mode === "Generative evolution" ? (
        <GenerationPlayer jobId={jobId} />
      ) : (
        <>
          <div className="inspection-toolbar">
            <label>
              <input
                type="checkbox"
                checked={locked}
                onChange={(e) => setLocked(e.target.checked)}
              />{" "}
              Lock selected point
            </label>
            {problem.grid.shape.map((n, i) => (
              <label key={i}>
                {["x", "y", "z"][i]} native index
                <input
                  aria-label={`Inspection ${["x", "y", "z"][i]} index`}
                  type="number"
                  min={0}
                  max={n - 1}
                  value={point[i]}
                  onChange={(e) =>
                    setPoint((p) =>
                      p.map((k, j) =>
                        i === j
                          ? Math.max(0, Math.min(n - 1, Number(e.target.value)))
                          : k,
                      ),
                    )
                  }
                />
              </label>
            ))}
            <label>
              <input
                type="checkbox"
                checked={surface}
                onChange={(e) => setSurface(e.target.checked)}
              />{" "}
              Height surface
            </label>
            <label>
              <input
                type="checkbox"
                checked={truth}
                onChange={(e) => setTruth(e.target.checked)}
              />{" "}
              Reference and error
            </label>
          </div>
          {mode === "Physical evolution" &&
            (problem.grid.temporal_kind === "static" ? (
              <p>
                Static Darcy has no physical-time animation. Use generation
                progress to inspect sampling.
              </p>
            ) : (
              <div className="playback">
                <Button onClick={() => setPlaying((v) => !v)}>
                  {playing ? "Pause physical time" : "Play physical time"}
                </Button>
                <label>
                  Physical time τ
                  <input
                    aria-label="Physical evolution time"
                    type="range"
                    min={0}
                    max={problem.grid.times.length - 1}
                    value={time}
                    onChange={(e) => onTime(Number(e.target.value))}
                  />
                </label>
                <strong>τ={problem.grid.times[time]}</strong>
              </div>
            ))}
          {error && <p role="alert">{error}</p>}
          {data && (
            <>
              <div className="linked-views">
                {data.views.map((view, i) => (
                  <div key={i} className="linked-plane">
                    {data.views.length > 1 && (
                      <h3>
                        Slice normal {["x", "y", "z"][i]} ={" "}
                        {view.slice_coordinate}
                      </h3>
                    )}
                    <div className="linked-pair">
                      <LinkedPlot
                        view={view}
                        values={view.values}
                        title="Reconstruction"
                        point={point}
                        onPoint={changePoint}
                      />
                      {surface && view.shape.length === 2 && (
                        <LinkedPlot
                          view={view}
                          values={view.values}
                          title="Reconstruction surface"
                          point={point}
                          onPoint={changePoint}
                          surface
                        />
                      )}
                    </div>
                    {truth && view.reference && (
                      <div className="linked-pair">
                        <LinkedPlot
                          view={view}
                          values={view.reference}
                          title="Reference"
                          point={point}
                          onPoint={changePoint}
                        />
                        <LinkedPlot
                          view={view}
                          values={view.error!}
                          title="Absolute error"
                          point={point}
                          onPoint={changePoint}
                          error
                        />
                      </div>
                    )}
                  </div>
                ))}
              </div>
              <div className="local-explanation">
                <div>
                  <span className="eyebrow">AT THE SELECTED POINT</span>
                  <p>
                    {Object.entries(data.coordinate)
                      .map(([a, v]) => `${a}=${v.toPrecision(5)}`)
                      .join(" · ")}
                    {data.physical_time !== null
                      ? ` · τ=${data.physical_time}`
                      : " · static"}
                  </p>
                  <div className="local-quantities">
                    {Object.entries({
                      ...data.diagnostics.fields,
                      ...data.diagnostics.terms,
                    }).map(([name, v]) => (
                      <span key={name}>
                        <i>{name}</i>
                        <strong>{v.toPrecision(5)}</strong>
                      </span>
                    ))}
                  </div>
                </div>
                <p>
                  {data.diagnostics.meaning}.{" "}
                  {data.observations_at_point.length
                    ? data.observations_at_point
                        .map(
                          (o) =>
                            `Measured ${o.measured.toPrecision(4)}, predicted ${o.predicted.toPrecision(4)}, residual ${o.residual.toPrecision(4)}.`,
                        )
                        .join(" ")
                    : "This point is unmeasured; its value is inferred."}{" "}
                  {data.reference_value === null
                    ? "No reference is attached; reference-error metrics are unavailable."
                    : "Reference is evaluation-only."}
                </p>
                {Boolean(data.diagnostics.flux) && (
                  <details>
                    <summary>Actual face fluxes</summary>
                    <pre>{JSON.stringify(data.diagnostics.flux, null, 2)}</pre>
                  </details>
                )}
                <Button
                  variant="outline"
                  onClick={() => saveFile("inspection-slice.json", data)}
                >
                  Download numerical slice & terms
                </Button>
                <Button
                  variant="ghost"
                  onClick={() => {
                    const svg = document.querySelector(
                      ".mathematical-workspace svg",
                    );
                    if (svg)
                      saveFile(
                        "field-view.svg",
                        svg.outerHTML.includes("xmlns=")
                          ? svg.outerHTML
                          : svg.outerHTML.replace(
                              "<svg ",
                              '<svg xmlns="http://www.w3.org/2000/svg" ',
                            ),
                      );
                  }}
                >
                  Download figure SVG
                </Button>
              </div>
            </>
          )}
        </>
      )}
    </section>
  );
}
