/* eslint-disable react-hooks/set-state-in-effect -- Accumulate externally polled job events and reset only when the selected job changes. */
import { useEffect, useState } from "react";
import type { Job } from "./types";
import { LineChart } from "./ScienceCharts";
type Recorded = {
  s: number;
  ess?: number;
  after?: number;
  resampled: boolean;
  entries: Record<string, number>;
  label: string;
};
export function LiveDiagnostics({ job }: { job: Job }) {
  const [trace, setTrace] = useState<Recorded[]>([]),
    [now, setNow] = useState(0);
  useEffect(() => {
    setNow(Date.now() / 1000);
    const timer = setInterval(() => setNow(Date.now() / 1000), 1000);
    return () => clearInterval(timer);
  }, []);
  useEffect(() => {
    setTrace([]);
  }, [job.id]);
  useEffect(() => {
    const event = job.process_event as
      | {
          decision?: { generation_position?: number };
          fk?: {
            generation_position: number;
            before: { ess: number[] };
            after: { ess: number[] };
            resampled_problems: boolean[];
          };
          report?: {
            label: string;
            entries: Record<string, { value: number | null }>;
          };
        }
      | undefined;
    const fk = event?.fk,
      s = fk?.generation_position ?? event?.decision?.generation_position;
    if (s === undefined) return;
    const row: Recorded = {
      s,
      ess: fk?.before.ess[0],
      after: fk?.after.ess[0],
      resampled: fk?.resampled_problems.some(Boolean) ?? false,
      entries: Object.fromEntries(
        Object.entries(event?.report?.entries ?? {})
          .filter(
            ([key, e]) =>
              (key.startsWith("observation.") || key.startsWith("pde.")) &&
              typeof e.value === "number",
          )
          .map(([k, e]) => [k, e.value!]),
      ),
      label: event?.report?.label ?? "FK numerical population",
    };
    setTrace((current) =>
      current.some((p) => p.s === s) ? current : [...current.slice(-255), row],
    );
  }, [job.process_event]);
  const keys = [...new Set(trace.flatMap((p) => Object.keys(p.entries)))];
  return (
    <div className="live-diagnostics">
      <p>
        {job.summary
          ? `${job.summary.mode} · ${job.summary.device} · grid ${job.summary.grid.join(" × ")} · checkpoint ${job.summary.model_id.slice(0, 12)}`
          : `Checkpoint ${job.model_id.slice(0, 12)}`}
      </p>
      <p>
        Elapsed{" "}
        {Math.max(0, (job.finished_at ?? now) - job.created_at).toFixed(1)}{" "}
        seconds · {job.stage}. Progress counts actual completed generation
        intervals; it is not an ETA.
      </p>
      {trace.some((p) => p.ess !== undefined) && (
        <LineChart
          title="Live FK effective sample size"
          xlabel="Generation coordinate s"
          ylabel="Effective particles"
          series={[
            {
              name: "ESS before weighting/resampling decision",
              color: "#426eab",
              points: trace
                .filter((p) => p.ess !== undefined)
                .map((p) => ({ x: p.s, y: p.ess! })),
            },
            {
              name: "Systematic resampling events (after ESS)",
              color: "#b47639",
              points: trace
                .filter((p) => p.resampled && p.after !== undefined)
                .map((p) => ({ x: p.s, y: p.after! })),
            },
          ]}
        />
      )}{" "}
      {keys.map((key) => (
        <LineChart
          key={key}
          title={key}
          xlabel="Generation coordinate s"
          ylabel="Estimated endpoint diagnostic"
          series={[
            {
              name: "Observed live endpoint estimates",
              color: "#426eab",
              points: trace
                .filter((p) => key in p.entries)
                .map((p) => ({ x: p.s, y: p.entries[key] })),
            },
          ]}
        />
      ))}
      <p className="small-text">
        Only diagnostics received during this view are plotted. No
        missing/future values are filled. Final returned-field verification is
        separate.
      </p>
    </div>
  );
}
