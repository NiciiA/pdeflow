import type { Point } from "./heat";
import { temporalPayload } from "./measurementTimeConfig";
import type { TimeControls } from "./measurementTimeConfig";
export function MeasurementTimes({
  mode,
  onMode,
  value,
  onChange,
  times,
  onTimes,
  total,
  support,
  manual,
}: {
  mode: string;
  onMode: (v: string) => void;
  value: TimeControls;
  onChange: (v: TimeControls) => void;
  times: string;
  onTimes: (v: string) => void;
  total: number;
  support: number;
  manual: boolean;
}) {
  const set = (k: keyof TimeControls, v: number | boolean | string) =>
    onChange({ ...value, [k]: v });
  const input = (
    label: string,
    k: "start" | "end" | "manual" | "early" | "initial",
    max = support,
  ) => (
    <label>
      {label}
      <input
        aria-label={label}
        type="number"
        min={0}
        max={max}
        step="any"
        value={value[k]}
        onChange={(e) => set(k, +e.target.value)}
      />
    </label>
  );
  const p = temporalPayload(mode, times, value);
  const resolved = p.time_counts.length
    ? p.time_counts.reduce((a, b) => a + b, 0)
    : total;
  return (
    <div className="measurement-times">
      <h3>Measurement times</h3>
      {manual ? (
        <>
          {input("Active measurement time", "manual")}
          <small>
            New clicks retain this timestamp. Changing display time does not
            move existing measurements in time.
          </small>
        </>
      ) : (
        <>
          <label>
            Measurement-time policy
            <select
              aria-label="Measurement-time policy"
              value={mode}
              onChange={(e) => onMode(e.target.value)}
            >
              <option value="continuous">
                Random across the time interval
              </option>
              <option value="chosen">Choose measurement times</option>
            </select>
          </label>
          {mode === "continuous" ? (
            <>
              {input("Measurement start time", "start")}
              {input("Measurement end time", "end")}
              <small>
                Randomized continuous times, stratified across the interval and
                shuffled. Total count is shared across all times.
              </small>
              <details>
                <summary>Advanced temporal mixture</summary>
                <label>
                  <input
                    type="checkbox"
                    checked={value.mixture}
                    onChange={(e) => set("mixture", e.target.checked)}
                  />
                  Extra early / exact initial-time coverage
                </label>
                {value.mixture && (
                  <>
                    {input("Early-time fraction", "early", 1)}
                    {input("Exact t=0 fraction", "initial", 1)}
                    <small>
                      Whole interval:{" "}
                      {(1 - value.early - value.initial).toFixed(3)} (must be
                      positive). Early times are log-uniform from 0.0001 to 0.2
                      of the selected interval.
                    </small>
                  </>
                )}
              </details>
            </>
          ) : (
            <>
              <label>
                Chosen physical times
                <input
                  aria-label="Chosen physical times"
                  value={times}
                  onChange={(e) => onTimes(e.target.value)}
                  placeholder="0.02, 0.2, 0.47, 0.83"
                />
              </label>
              <label>
                Optional counts per time
                <input
                  aria-label="Counts per measurement time"
                  value={value.counts}
                  onChange={(e) => set("counts", e.target.value)}
                  placeholder="Leave blank to distribute the total"
                />
              </label>
              <small>
                Blank counts distribute the selected total across the entered
                times; counts specify the total explicitly.
              </small>
            </>
          )}
          <p>
            Resolved total: <b>{resolved}</b> measurements · supported time 0–
            {support}. Apply with Take measurements; existing measurements
            remain unchanged.
          </p>
        </>
      )}
    </div>
  );
}
export function MeasurementSummary({
  rows,
  policy,
  end,
}: {
  rows: Point[];
  policy: string;
  end: number;
}) {
  if (!rows.length) return null;
  const bins = Array<number>(8).fill(0);
  for (const r of rows)
    bins[Math.min(7, Math.max(0, Math.floor((r.time / end) * 8)))]++;
  return (
    <div aria-label="Measurement time summary">
      <p>
        {rows.length} measurements · t ∈ [
        {Math.min(...rows.map((r) => r.time)).toPrecision(6)},{" "}
        {Math.max(...rows.map((r) => r.time)).toPrecision(6)}] · {policy}
      </p>
      <div
        role="img"
        aria-label={`Measurement time histogram: ${bins.join(", ")}`}
        style={{ display: "flex", gap: 3, height: 35, alignItems: "end" }}
      >
        {bins.map((n, i) => (
          <span
            key={i}
            title={`${(i * end) / 8}–${((i + 1) * end) / 8}: ${n}`}
            style={{
              background: "#588e95",
              flex: 1,
              height: `${Math.max(2, (100 * n) / Math.max(...bins))}%`,
            }}
          />
        ))}
      </div>
      <small>
        Timeline: 0 to {end}. Faded markers indicate temporal distance only, not
        a change in value or uncertainty.
      </small>
      <details>
        <summary>Inspect sensor timestamps and values</summary>
        <div style={{ maxHeight: 240, overflow: "auto" }}>
          {rows.map((r, i) => (
            <article
              key={i}
              style={{
                borderBottom: "1px solid #dce5e4",
                padding: "8px 0",
                overflowWrap: "anywhere",
              }}
              aria-label={`Sensor ${i + 1}`}
            >
              <b>
                Sensor {i + 1} · {r.field}
              </b>
              <div>
                Physical time:{" "}
                <span data-testid={`sensor-time-${i}`}>{r.time}</span>
              </div>
              <div>x: {r.x}</div>
              <div>y: {r.y}</div>
              <div>
                Value: {r.value} · σ: {r.sigma}
              </div>
            </article>
          ))}
        </div>
      </details>
    </div>
  );
}
