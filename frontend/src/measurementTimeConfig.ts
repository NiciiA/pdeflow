export type TimeControls = {
  start: number;
  end: number;
  manual: number;
  mixture: boolean;
  early: number;
  initial: number;
  counts: string;
};
export function temporalPayload(
  pattern: string,
  times: string,
  c: TimeControls,
) {
  const entered = times
    .split(/[,\s]+/)
    .filter(Boolean)
    .map(Number);
  const counts = c.counts.trim()
    ? c.counts
        .split(/[,\s]+/)
        .filter(Boolean)
        .map(Number)
    : [];
  return {
    start: c.start,
    end: c.end,
    times: pattern === "chosen" ? entered : [],
    time_counts: pattern === "chosen" ? counts : [],
    temporal_policy: c.mixture
      ? {
          policy: "continuous_mixture",
          whole: 1 - c.early - c.initial,
          early: c.early,
          initial: c.initial,
        }
      : { policy: "continuous_uniform" },
  };
}
