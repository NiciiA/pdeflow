import type { Marginals, ProfileMode, ProfileScales } from "./heatProfiles";
import { profilePoints } from "./heatProfiles";
export type ProfileHover = {
  axis: "x" | "y";
  fraction: number;
  otherFraction?: number;
} | null;
export function SpatialProfile({
  fieldSymbol = "u",
  profileLabel = "Heat",
  axis,
  profiles,
  scales,
  mode,
  label,
  origin,
  length,
  hover,
  onHover,
}: {
  fieldSymbol?: string;
  profileLabel?: string;
  axis: "x" | "y";
  profiles: Marginals;
  scales: ProfileScales;
  mode: ProfileMode;
  label: string;
  origin: number;
  length: number;
  hover: ProfileHover;
  onHover: (value: ProfileHover) => void;
}) {
  const values = profiles[axis],
    [lo, hi] = scales[axis];
  const map = (value: number) => 1000 * (1 - (value - lo) / (hi - lo));
  const zero = map(0),
    points = profilePoints(values);
  const curve = points.map(({ fraction, value }) =>
    axis === "x"
      ? `${fraction * 1000},${map(value)}`
      : `${map(value)},${(1 - fraction) * 1000}`,
  );
  const fill =
    axis === "x"
      ? `M0,${zero} L${curve.join(" L")} L1000,${zero} Z`
      : `M${zero},1000 L${curve.join(" L")} L${zero},0 Z`;
  const heading = `${mode === "magnitude" ? "Magnitude" : profileLabel} along ${axis}`;
  const active = hover
    ? hover.axis === axis
      ? hover.fraction
      : (hover.otherFraction ?? null)
    : null;
  const pos = active === null ? 0 : active * values.length;
  const index = Math.floor(pos) % values.length,
    w = pos - Math.floor(pos);
  const value =
    values[index] * (1 - w) + values[(index + 1) % values.length] * w;
  return (
    <div
      className={`spatial-profile profile-${axis}`}
      data-scale={JSON.stringify([lo, hi])}
    >
      <svg
        role="img"
        tabIndex={0}
        onFocus={() => onHover({ axis, fraction: 0.5 })}
        onBlur={() => onHover(null)}
        onKeyDown={(e) => {
          if (
            ["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown"].includes(e.key)
          ) {
            e.preventDefault();
            const direction =
              e.key === "ArrowRight" || e.key === "ArrowUp" ? 1 : -1;
            onHover({
              axis,
              fraction: Math.max(
                0,
                Math.min(1, (active ?? 0.5) + direction / values.length),
              ),
            });
          }
        }}
        aria-label={`${label}: ${heading}`}
        viewBox="0 0 1000 1000"
        preserveAspectRatio="none"
        onPointerMove={(e) => {
          const r = e.currentTarget.getBoundingClientRect();
          onHover({
            axis,
            fraction: Math.max(
              0,
              Math.min(
                1,
                axis === "x"
                  ? (e.clientX - r.left) / r.width
                  : 1 - (e.clientY - r.top) / r.height,
              ),
            ),
          });
        }}
        onPointerLeave={() => onHover(null)}
      >
        <title>
          {heading}. Field values integrated across the other spatial direction.
        </title>
        {[0.25, 0.5, 0.75].map((f) => (
          <line
            key={f}
            className="profile-gridline"
            x1={axis === "x" ? f * 1000 : 0}
            x2={axis === "x" ? f * 1000 : 1000}
            y1={axis === "y" ? f * 1000 : 0}
            y2={axis === "y" ? f * 1000 : 1000}
          />
        ))}
        <path className="profile-fill" d={fill} />
        <line
          className="profile-zero"
          x1={axis === "x" ? 0 : zero}
          x2={axis === "x" ? 1000 : zero}
          y1={axis === "x" ? zero : 0}
          y2={axis === "x" ? zero : 1000}
        />
        <path className="profile-curve" d={`M${curve.join(" L")}`} />
        {active !== null && (
          <circle
            className="profile-dot"
            r="4"
            vectorEffect="non-scaling-stroke"
            cx={axis === "x" ? active * 1000 : map(value)}
            cy={axis === "x" ? map(value) : (1 - active) * 1000}
          />
        )}
      </svg>
      <span
        className="profile-zero-label"
        style={
          axis === "x"
            ? { top: `${zero / 10}%`, left: 2 }
            : { left: `${zero / 10}%`, bottom: 2 }
        }
      >
        0
      </span>
      <span
        className="profile-heading"
        title="Field values integrated across the other spatial direction."
      >
        {heading}
      </span>
      <span className="profile-extent">
        {lo.toPrecision(2)} … {hi.toPrecision(2)}
      </span>
      {active !== null && (
        <output className="profile-readout">
          {axis} = {(origin + active * length).toPrecision(3)} · ∫
          {mode === "magnitude" ? `|${fieldSymbol}|` : fieldSymbol} ={" "}
          {value.toPrecision(4)}
        </output>
      )}
    </div>
  );
}
