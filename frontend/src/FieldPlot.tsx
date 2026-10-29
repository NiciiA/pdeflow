import type { Row, Slice } from "./types";
function color(t: number) {
  const v = Math.max(0, Math.min(1, t));
  return `rgb(${Math.round(40 + 210 * v)},${Math.round(80 + 105 * (1 - Math.abs(2 * v - 1)))},${Math.round(210 - 160 * v)})`;
}
export function FieldPlot({
  slice,
  values,
  title,
  observations = false,
}: {
  slice: Slice;
  values: number[] | number[][] | null;
  title: string;
  observations?: boolean;
}) {
  if (values === null)
    return (
      <section className="plot">
        <h3>{title}</h3>
        <p>No evaluation reference attached.</p>
      </section>
    );
  const rows = slice.shape.length === 1 ? 1 : slice.shape[1],
    cols = slice.shape[0],
    flat = (values as number[][]).flat();
  const low = title === "Absolute error" ? 0 : slice.vmin,
    high = title === "Absolute error" ? Math.max(...flat) : slice.vmax;
  const x = slice.axes[0],
    y = slice.axes[1];
  const coords = slice.coordinates[x];
  const yy = y ? slice.coordinates[y] : [];
  const px = (value: number) =>
    45 +
    ((value - (coords[0] ?? 0)) /
      ((coords.at(-1) ?? 1) - (coords[0] ?? 0) || 1)) *
      300;
  const points = observations ? slice.observations : [];
  return (
    <section className="plot">
      <h3>{title}</h3>
      <svg
        viewBox="0 0 390 285"
        role="img"
        aria-label={`${title}, ${slice.axes.join(" / ")}, ${slice.physical_time === null ? "static field" : `physical time ${slice.physical_time}`}`}
      >
        {slice.shape.length === 1 ? (
          <>
            <polyline
              fill="none"
              stroke={title === "Reference" ? "#775994" : "#247c75"}
              strokeDasharray={title === "Reference" ? "6 4" : undefined}
              strokeWidth="2"
              points={(values as number[])
                .map(
                  (v, i) =>
                    `${px(coords[i])},${220 - (170 * (v - low)) / (high - low || 1)}`,
                )
                .join(" ")}
            />
            {points.map((p: Row, i) => (
              <circle
                key={i}
                cx={px(Number(p[x]))}
                cy={220 - (170 * (Number(p.value) - low)) / (high - low || 1)}
                r="4"
                fill="#d57b38"
              />
            ))}
          </>
        ) : (
          Array.from({ length: cols }, (_, i) =>
            Array.from({ length: rows }, (_, j) => (
              <rect
                key={`${i}-${j}`}
                x={45 + (i * 300) / cols}
                y={25 + ((rows - j - 1) * 200) / rows}
                width={301 / cols}
                height={201 / rows}
                fill={color(
                  ((values as number[][])[i][j] - low) / (high - low || 1),
                )}
              />
            )),
          )
        )}
        {slice.shape.length > 1 &&
          points.map((p, i) => (
            <circle
              key={i}
              cx={
                45 +
                (((Number(p[x]) - coords[0]) /
                  (coords.at(-1)! - coords[0] || 1)) *
                  300 *
                  (cols - 1)) /
                  cols +
                150 / cols
              }
              cy={
                25 +
                200 -
                (((Number(p[y]) - yy[0]) / (yy.at(-1)! - yy[0] || 1)) *
                  200 *
                  (rows - 1)) /
                  rows -
                100 / rows
              }
              r="4"
              stroke="white"
              fill="#d57b38"
            />
          ))}
        <line x1="45" y1="226" x2="345" y2="226" stroke="currentColor" />
        <text x={slice.shape.length === 1 ? 45 : 45 + 150 / cols} y="245">
          {coords[0]?.toPrecision(3)}
        </text>
        <text x={slice.shape.length === 1 ? 310 : 325 - 150 / cols} y="245">
          {coords.at(-1)?.toPrecision(3)}
        </text>
        <text x="180" y="269">
          {x} [{slice.units}]
        </text>
        {y && (
          <g>
            <text transform="translate(12,130) rotate(-90)">{y}</text>
            <text x="41" textAnchor="end" y={225 - 100 / rows}>
              {yy[0]?.toPrecision(3)}
            </text>
            <text x="41" textAnchor="end" y={25 + 100 / rows}>
              {yy.at(-1)?.toPrecision(3)}
            </text>
          </g>
        )}
      </svg>
      {slice.display_downsampled && (
        <p className="plot-scale">
          Display {slice.shape.join("×")} sampled from native numerical grid{" "}
          {slice.native_shape?.join("×")}; display striding does not change
          model/reference resolution.
        </p>
      )}
      <p className="plot-scale">
        {low.toPrecision(3)} — {high.toPrecision(3)} ·{" "}
        {observations
          ? "orange dots: supplied observations"
          : title === "Empirical spread" || title === "Absolute error"
            ? "independent diagnostic scale"
            : "shared physical field scale"}
      </p>
    </section>
  );
}
