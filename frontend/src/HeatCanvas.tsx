import { useEffect, useMemo, useRef, useState } from "react";
import type { PointerEvent } from "react";
import type { Point } from "./heat";
import { spatialMarginals, marginalScales } from "./heatProfiles";
import type { ProfileMode, ProfileScales } from "./heatProfiles";
import { SpatialProfile } from "./SpatialProfile";
import type { ProfileHover } from "./SpatialProfile";
const colors = [
  [13, 24, 47],
  [35, 61, 91],
  [34, 108, 124],
  [66, 156, 151],
  [155, 202, 174],
  [246, 219, 167],
  [249, 147, 96],
  [211, 69, 63],
];
export function HeatCanvas({
  values,
  range,
  label,
  points = [],
  time = 0,
  lengths = [1, 1],
  origins = [0, 0],
  onPoint,
  onFinish,
  candidates = [],
  discrete = false,
  showProfiles = false,
  profileMode = "signed",
  profileValues,
  profileScales,
  regions,
  fieldSymbol = "u",
  profileLabel = "Heat",
}: {
  fieldSymbol?: string;
  profileLabel?: string;
  values: number[][];
  range: number[];
  label: string;
  points?: Point[];
  time?: number;
  lengths?: number[];
  origins?: number[];
  onPoint?: (x: number, y: number) => void;
  onFinish?: () => void;
  candidates?: number[][];
  discrete?: boolean;
  showProfiles?: boolean;
  profileMode?: ProfileMode;
  profileValues?: number[][];
  profileScales?: ProfileScales;
  regions?: number[][];
}) {
  const canvas = useRef<HTMLCanvasElement>(null),
    down = useRef(false);
  const [hover, setHover] = useState<ProfileHover>(null);
  const profiles = useMemo(
    () => spatialMarginals(profileValues || values, lengths, profileMode),
    [profileValues, values, lengths, profileMode],
  );
  const scales =
    profileScales ||
    marginalScales(
      [profiles],
      Math.max(...range.map(Math.abs)) * Math.max(...lengths) * 1e-12,
    );
  useEffect(() => {
    const c = canvas.current;
    if (!c) return;
    const draw = () => {
      const box = c.getBoundingClientRect(),
        dpr = window.devicePixelRatio || 1;
      c.width = Math.round(box.width * dpr);
      c.height = Math.round(box.height * dpr);
      const ctx = c.getContext("2d");
      if (!ctx) return;
      const pixel = document.createElement("canvas");
      pixel.width = values.length;
      pixel.height = values[0].length;
      const pc = pixel.getContext("2d")!;
      const im = pc.createImageData(pixel.width, pixel.height);
      for (let x = 0; x < pixel.width; x++)
        for (let y = 0; y < pixel.height; y++) {
          const f =
              Math.max(
                0,
                Math.min(
                  1,
                  (values[x][y] - range[0]) / (range[1] - range[0] || 1),
                ),
              ) *
              (colors.length - 1),
            k = Math.min(colors.length - 2, Math.floor(f)),
            w = f - k;
          const ix = ((pixel.height - 1 - y) * pixel.width + x) * 4;
          for (let q = 0; q < 3; q++)
            im.data[ix + q] = colors[k][q] * (1 - w) + colors[k + 1][q] * w;
          im.data[ix + 3] = 255;
        }
      pc.putImageData(im, 0, 0);
      ctx.imageSmoothingEnabled = false;
      // Periodic node i is at i/N, not the centre of display cell i.
      // Shift half a cell and wrap the image so profiles/markers share its coordinates.
      const dx = c.width / pixel.width,
        dy = c.height / pixel.height;
      for (const sx of [-1, 0, 1])
        for (const sy of [-1, 0, 1])
          ctx.drawImage(
            pixel,
            sx * c.width - dx / 2,
            sy * c.height + dy / 2,
            c.width,
            c.height,
          );
      if (regions) {
        const nx = regions.length,
          ny = regions[0].length,
          seen = new Set<number>();
        ctx.strokeStyle = "#f7e9c9";
        ctx.lineWidth = dpr;
        ctx.setLineDash([3 * dpr, 2 * dpr]);
        for (let i = 0; i < nx; i++)
          for (let j = 0; j < ny; j++) {
            const id = regions[i][j];
            if (!id) continue;
            const x = (i / nx) * c.width,
              y = (1 - j / ny) * c.height,
              dx = c.width / nx,
              dy = c.height / ny;
            for (const [di, dj] of [
              [1, 0],
              [-1, 0],
              [0, 1],
              [0, -1],
            ])
              if (regions[(i + di + nx) % nx][(j + dj + ny) % ny] !== id) {
                ctx.beginPath();
                if (di) {
                  ctx.moveTo(x + (di * dx) / 2, y - dy / 2);
                  ctx.lineTo(x + (di * dx) / 2, y + dy / 2);
                } else {
                  ctx.moveTo(x - dx / 2, y - (dj * dy) / 2);
                  ctx.lineTo(x + dx / 2, y - (dj * dy) / 2);
                }
                ctx.stroke();
              }
            if (!seen.has(id)) {
              seen.add(id);
              ctx.fillStyle = "#fff";
              ctx.font = `${10 * dpr}px sans-serif`;
              ctx.fillText(
                String(id),
                Math.min(c.width - 14 * dpr, Math.max(4 * dpr, x)),
                Math.min(c.height - 3 * dpr, Math.max(12 * dpr, y)),
              );
            }
          }
        ctx.setLineDash([]);
      }
      for (const [px, py] of candidates) {
        const x = ((px - origins[0]) / lengths[0]) * c.width,
          y = (1 - (py - origins[1]) / lengths[1]) * c.height;
        ctx.beginPath();
        ctx.arc(x, y, 5 * dpr, 0, 2 * Math.PI);
        ctx.strokeStyle = "#9343b5";
        ctx.lineWidth = 2 * dpr;
        ctx.setLineDash([3 * dpr, 2 * dpr]);
        ctx.stroke();
        ctx.setLineDash([]);
      }
      for (const p of points.filter((p) => Math.abs(p.time - time) < 1e-7)) {
        const x = ((p.x - origins[0]) / lengths[0]) * c.width,
          y = (1 - (p.y - origins[1]) / lengths[1]) * c.height;
        ctx.beginPath();
        ctx.arc(x, y, 3.5 * dpr, 0, 2 * Math.PI);
        ctx.fillStyle = "#fff";
        ctx.fill();
        ctx.lineWidth = 1.4 * dpr;
        ctx.strokeStyle = "#172435";
        ctx.stroke();
      }
    };
    draw();
    const resize = new ResizeObserver(draw);
    resize.observe(c);
    return () => resize.disconnect();
  }, [values, range, points, time, lengths, origins, candidates, regions]);
  function point(e: PointerEvent<HTMLCanvasElement>) {
    const r = e.currentTarget.getBoundingClientRect();
    onPoint?.(
      Math.max(0, Math.min(0.999999, (e.clientX - r.left) / r.width)),
      Math.max(0, Math.min(0.999999, 1 - (e.clientY - r.top) / r.height)),
    );
  }
  return (
    <figure
      className="heat-figure"
      style={{
        maxWidth: `calc(min(52vh, 520px) * ${(lengths[0] / lengths[1]) * (showProfiles ? 1.2 : 1)})`,
      }}
    >
      <div className={`heat-plot ${showProfiles ? "with-profiles" : ""}`}>
        {showProfiles && (
          <SpatialProfile
            fieldSymbol={fieldSymbol}
            profileLabel={profileLabel}
            axis="y"
            profiles={profiles}
            scales={scales}
            mode={profileMode}
            label={label}
            origin={origins[1]}
            length={lengths[1]}
            hover={hover}
            onHover={setHover}
          />
        )}
        <div
          className="heatmap-surface"
          style={{ aspectRatio: `${lengths[0]} / ${lengths[1]}` }}
        >
          <canvas
            ref={canvas}
            role="img"
            aria-label={label}
            className={onPoint ? "paintable" : ""}
            onPointerDown={(e) => {
              if (!onPoint) return;
              setHover(null);
              down.current = true;
              e.currentTarget.setPointerCapture(e.pointerId);
              point(e);
            }}
            onPointerMove={(e) => {
              if (down.current && !discrete) point(e);
              else if (showProfiles && !onPoint) {
                const r = e.currentTarget.getBoundingClientRect();
                setHover({
                  axis: "x",
                  fraction: (e.clientX - r.left) / r.width,
                  otherFraction: 1 - (e.clientY - r.top) / r.height,
                });
              }
            }}
            onPointerLeave={() => setHover(null)}
            onPointerUp={() => {
              if (down.current) {
                down.current = false;
                onFinish?.();
              }
            }}
            onPointerCancel={() => {
              down.current = false;
              onFinish?.();
            }}
          />
          {showProfiles && hover && (
            <svg
              className="profile-crosshair"
              viewBox="0 0 1000 1000"
              preserveAspectRatio="none"
              aria-hidden="true"
            >
              <line
                x1={hover.axis === "x" ? hover.fraction * 1000 : 0}
                x2={hover.axis === "x" ? hover.fraction * 1000 : 1000}
                y1={hover.axis === "y" ? (1 - hover.fraction) * 1000 : 0}
                y2={hover.axis === "y" ? (1 - hover.fraction) * 1000 : 1000}
              />
              {hover.otherFraction !== undefined && (
                <line
                  x1="0"
                  x2="1000"
                  y1={(1 - hover.otherFraction) * 1000}
                  y2={(1 - hover.otherFraction) * 1000}
                />
              )}
            </svg>
          )}
        </div>
        {showProfiles && (
          <>
            <span
              className="profile-corner"
              title="Field values integrated across the other spatial direction."
            >
              ∫ {profileMode === "magnitude" ? `|${fieldSymbol}|` : fieldSymbol}
            </span>
            <SpatialProfile
              fieldSymbol={fieldSymbol}
              profileLabel={profileLabel}
              axis="x"
              profiles={profiles}
              scales={scales}
              mode={profileMode}
              label={label}
              origin={origins[0]}
              length={lengths[0]}
              hover={hover}
              onHover={setHover}
            />
          </>
        )}
      </div>
      <figcaption>
        {label}
        <span>
          {" "}
          x {origins[0]}–{origins[0] + lengths[0]} → · y {origins[1]}–
          {origins[1] + lengths[1]} ↑
        </span>
      </figcaption>
    </figure>
  );
}
export function ColorBar({
  range,
  label = "Heat u",
}: {
  range: number[];
  label?: string;
}) {
  return (
    <div
      className="heat-legend"
      aria-label={`${label}: ${range[0]} to ${range[1]}`}
    >
      <span>{range[0].toPrecision(3)}</span>
      <i />
      <span>{range[1].toPrecision(3)}</span>
      <b>{label}</b>
    </div>
  );
}
