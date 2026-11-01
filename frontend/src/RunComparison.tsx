import { useState } from "react";
import { api } from "./types";
import type { Job } from "./types";
import { Button } from "./components/ui/button";
import {
  Card,
  CardHeader,
  CardTitle,
  CardContent,
  CardDescription,
} from "./components/ui/card";
type Comparison = {
  reference_note: string;
  warning: string;
  runs: {
    id: string;
    mode: string;
    checkpoint: string;
    seconds: number | null;
    observation_rmse: number | null;
    pde_fd_rmse: number | null;
    relative_l2_unobserved_mean: number | null;
    stop_reason: string;
  }[];
};
export function RunComparison({ jobs }: { jobs: Job[] }) {
  const [ids, setIds] = useState<string[]>([]),
    [result, setResult] = useState<Comparison | null>(null),
    [error, setError] = useState("");
  const rows = result?.runs ?? [],
    finite = rows.filter(
      (r) => r.seconds !== null && r.observation_rmse !== null,
    ),
    xmax = Math.max(...finite.map((r) => r.seconds!), 0.001),
    ymax = Math.max(...finite.map((r) => r.observation_rmse!), 0.001);
  return (
    <Card>
      <CardHeader>
        <CardTitle>Compare the same problem</CardTitle>
        <CardDescription>
          Select2–4 completed runs. The server checks the actual equation, known
          inputs, grid, measurements and noise before plotting quality against
          measured cost.
        </CardDescription>
      </CardHeader>
      <CardContent>
        <div className="compare-options">
          {jobs
            .filter((j) => j.status === "succeeded")
            .slice(0, 30)
            .map((j) => (
              <label key={j.id}>
                <input
                  aria-label={`Compare run ${j.id.slice(0, 8)}`}
                  type="checkbox"
                  checked={ids.includes(j.id)}
                  disabled={!ids.includes(j.id) && ids.length >= 4}
                  onChange={(e) => {
                    setIds(
                      e.target.checked
                        ? [...ids, j.id]
                        : ids.filter((id) => id !== j.id),
                    );
                    setResult(null);
                  }}
                />
                {j.id.slice(0, 8)} · {j.stop_reason || "execution finished"}
              </label>
            ))}
        </div>
        <Button
          variant="outline"
          disabled={ids.length < 2}
          onClick={() =>
            void api<Comparison>("/compare", { ids })
              .then((v) => {
                setResult(v);
                setError("");
              })
              .catch((e) => {
                setError(String(e));
                setResult(null);
              })
          }
        >
          Compare selected runs
        </Button>
        {error && <p role="alert">{error}</p>}
        {result && (
          <>
            <figure className="science-chart">
              <h3>Observation agreement versus measured inference time</h3>
              <svg
                viewBox="0 0 430 280"
                role="img"
                aria-label="Compatible runs: observation RMSE versus inference seconds"
              >
                <path d="M55,25V225H390" fill="none" stroke="#8a9aaf" />
                {finite.map((r, i) => (
                  <g key={r.id}>
                    <circle
                      cx={55 + (300 * r.seconds!) / xmax}
                      cy={225 - (180 * r.observation_rmse!) / ymax}
                      r="6"
                      fill="#426eab"
                    />
                    <text
                      x={63 + (300 * r.seconds!) / xmax}
                      y={220 - (180 * r.observation_rmse!) / ymax}
                    >
                      {i + 1}
                    </text>
                  </g>
                ))}
                <text x="125" y="266">
                  Measured inference seconds
                </text>
                <text transform="translate(14,200) rotate(-90)">
                  Observation RMSE
                </text>
                <text x="55" y="243">
                  0
                </text>
                <text x="350" y="243">
                  {xmax.toPrecision(3)}
                </text>
                <text x="50" textAnchor="end" y="42">
                  {ymax.toPrecision(3)}
                </text>
              </svg>
              <figcaption>
                Numbered points correspond to rows below. Lower error and time
                are separate tradeoffs; methods can use different computation.
              </figcaption>
            </figure>
            <div className="bounded-table">
              <table aria-label="Compatible run comparison">
                <thead>
                  <tr>
                    <th>Run / method</th>
                    <th>Observation RMSE</th>
                    <th>PDE RMSE</th>
                    <th>Reference error</th>
                    <th>Seconds</th>
                    <th>Outcome</th>
                  </tr>
                </thead>
                <tbody>
                  {rows.map((r, i) => (
                    <tr key={r.id}>
                      <td>
                        {i + 1}. {r.id.slice(0, 8)} / {r.mode}
                        <small>{r.checkpoint?.slice(0, 12)}</small>
                      </td>
                      {[
                        r.observation_rmse,
                        r.pde_fd_rmse,
                        r.relative_l2_unobserved_mean,
                        r.seconds,
                      ].map((v, j) => (
                        <td key={j}>
                          {v === null ? "Unavailable" : v.toPrecision(4)}
                        </td>
                      ))}
                      <td>{r.stop_reason}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <p>
              {result.reference_note}. {result.warning}
            </p>
          </>
        )}
      </CardContent>
    </Card>
  );
}
