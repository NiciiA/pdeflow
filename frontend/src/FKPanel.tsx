import { Card, CardHeader, CardTitle, CardContent } from "./components/ui/card";
import { Input } from "./components/ui/input";
import { defaultFK } from "./fkConfig";

export function FKControls({
  value,
  onChange,
}: {
  value: unknown;
  onChange: (value: unknown) => void;
}) {
  const cfg = { ...defaultFK, ...((value as Record<string, unknown>) ?? {}) };
  const change = (name: string, v: unknown) => onChange({ ...cfg, [name]: v });
  return (
    <Card>
      <CardHeader>
        <CardTitle>Feynman–Kac population</CardTitle>
      </CardHeader>
      <CardContent>
        <p>
          Particles are candidate complete solutions, not physical molecules.
          Euler–Maruyama uses a score-corrected linear-CFM drift and noise that
          vanishes at the endpoint. No gradient proposal or refinement is
          combined with FK.
        </p>
        <p>
          Systematic resampling is eligible at fixed intervals and triggered by
          ESS. Terminal fields carry normalized weights unless final resampling
          is selected. Incomplete paths are not terminal-target samples.
          Particle-sampler resume is unavailable.
        </p>
        <div className="form-pair">
          {[
            "kappa",
            "resample_every",
            "ess_fraction",
            "chunk_size",
            "observation_weight",
            "physics_weight",
          ].map((name) => (
            <label key={name}>
              {name.replaceAll("_", " ")}
              <Input
                aria-label={"FK " + name}
                type="number"
                step="any"
                value={Number(cfg[name as keyof typeof cfg])}
                onChange={(e) => change(name, Number(e.target.value))}
              />
            </label>
          ))}
        </div>
        <label>
          Observation energy convention
          <select
            aria-label="FK observation convention"
            value={String(cfg.observation_convention)}
            onChange={(e) => change("observation_convention", e.target.value)}
          >
            <option value="gaussian_sum">
              Gaussian sum (weighted likelihood)
            </option>
            <option value="tempered_mean">
              Tempered mean (different target temperature)
            </option>
          </select>
        </label>
        <label>
          <input
            type="checkbox"
            aria-label="FK terminal resampling"
            checked={Boolean(cfg.terminal_resample)}
            onChange={(e) => change("terminal_resample", e.target.checked)}
          />{" "}
          Final systematic resampling for equal-weight output
        </label>
        <p>
          Fixed method: euler_maruyama · systematic · annealed_difference ·
          linear_vanish. The ODE controls apply to gradient/unguided modes.
          Current learned-prior errors and finite-step/particle errors remain;
          no exact physical posterior is claimed.
        </p>
      </CardContent>
    </Card>
  );
}
type Trace = {
  generation_position: number;
  before: { ess: number[]; max_weight: number[] };
  after: { ess: number[] };
  unique_ancestors: number[];
  resampled_problems: boolean[];
  trigger: string;
  energy_components_weighted_mean: Record<string, number[]>;
};
type Result = {
  trace: Trace[];
  terminal_weights: number[][];
  terminal_target_complete: boolean;
  model_evaluations: number;
  model_particle_evaluations: number;
  energy_particle_evaluations: number;
  config: { terminal_resample: boolean };
};
export function FKPanel({ metrics }: { metrics: Record<string, unknown> }) {
  const fk = (metrics.cost as { fk?: Result } | undefined)?.fk;
  if (!fk) return null;
  const values = fk.trace.map((t) => t.before.ess[0]);
  const max = Math.max(1, ...values);
  const points = fk.trace
    .map(
      (t) =>
        `${42 + t.generation_position * 530},${105 - (80 * t.before.ess[0]) / max}`,
    )
    .join(" ");
  return (
    <Card className="process-panel">
      <CardHeader>
        <CardTitle>FK population diagnostics</CardTitle>
      </CardHeader>
      <CardContent>
        <p aria-label="FK terminal status">
          {fk.terminal_target_complete
            ? "Terminal probability path completed"
            : "Incomplete path · not terminal-target samples"}
        </p>
        <p>
          {fk.config.terminal_resample
            ? "Final systematic resampling; equal-weight descendants"
            : "Normalized weighted output; means and variances respect weights"}
          . ESS is a weight diagnostic, not reconstruction accuracy or
          calibrated uncertainty.
        </p>
        <svg
          role="img"
          aria-label="Actual FK ESS trace"
          viewBox="0 0 600 160"
          style={{ width: "100%", maxWidth: 600 }}
        >
          <polyline
            fill="none"
            stroke="#19776c"
            strokeWidth="2"
            points={points}
          />
          {fk.trace.map((t, i) =>
            t.resampled_problems.some(Boolean) ? (
              <circle
                key={i}
                cx={42 + t.generation_position * 530}
                cy={105 - (80 * t.before.ess[0]) / max}
                r="4"
                fill="#9b4d12"
              />
            ) : null,
          )}
          <path d="M42,18V105H575" fill="none" stroke="#8a9aaf" />
          <text x="42" y="120" fontSize="10">
            0
          </text>
          <text x="565" y="120" fontSize="10">
            1
          </text>
          <text x="35" textAnchor="end" y="28" fontSize="10">
            {max.toPrecision(3)}
          </text>
          <text transform="translate(12,100) rotate(-90)" fontSize="11">
            Effective particles
          </text>
          <text x="100" y="140" fontSize="11">
            Generation coordinate s · dots mark systematic resampling
          </text>
        </svg>
        <table aria-label="FK population trace">
          <thead>
            <tr>
              <th>Position</th>
              <th>ESS before / after</th>
              <th>Max weight</th>
              <th>Root ancestors</th>
              <th>Resampling</th>
              <th>Weighted energy components</th>
            </tr>
          </thead>
          <tbody>
            {fk.trace.map((t, i) => (
              <tr key={i}>
                <td>{t.generation_position.toFixed(3)}</td>
                <td>
                  {t.before.ess[0].toFixed(2)} / {t.after.ess[0].toFixed(2)}
                </td>
                <td>{t.before.max_weight[0].toFixed(4)}</td>
                <td>{t.unique_ancestors[0]}</td>
                <td>{t.trigger}</td>
                <td>{JSON.stringify(t.energy_components_weighted_mean)}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <p>
          Model calls: {fk.model_evaluations}; particle model evaluations:{" "}
          {fk.model_particle_evaluations}; particle energy evaluations:{" "}
          {fk.energy_particle_evaluations}.
        </p>
        <details>
          <summary>Terminal particle weights</summary>
          <pre className="provenance">
            {JSON.stringify(fk.terminal_weights)}
          </pre>
        </details>
      </CardContent>
    </Card>
  );
}
