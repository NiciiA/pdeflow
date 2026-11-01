import type { Reconstruction } from "./heat";
export function ReconstructionDiagnostics({
  metrics,
}: {
  metrics: Reconstruction["metrics"];
}) {
  if (!metrics.sensor_diagnostics) return null;
  return (
    <details className="advanced-panel">
      <summary>Reconstruction diagnostics</summary>
      <p>
        The displayed field is the ensemble mean. Sensor errors below query each
        measurement at its exact coordinate, physical time and quantity; markers
        alone do not show agreement.
      </p>
      {Object.entries(metrics.sensor_diagnostics.by_quantity).map(
        ([quantity, row]) => (
          <p key={quantity}>
            <b>
              {quantity} · {row.count} sensors
            </b>
            {" · "}
            Ensemble MSE {row.ensemble_mean.mse.toPrecision(4)} · RMSE{" "}
            {row.ensemble_mean.rmse.toPrecision(4)} · MAE{" "}
            {row.ensemble_mean.mae.toPrecision(4)}.{" "}
            {row.ensemble_mean.noiseless_count} noiseless /{" "}
            {row.ensemble_mean.noisy_count} noisy measurements. Physical units;
            this is conditioning-data fit, not held-out accuracy.
          </p>
        ),
      )}
      <p>
        Checkpoint{" "}
        <code>{metrics.checkpoint_sha256 || "identity unavailable"}</code> ·{" "}
        {metrics.checkpoint_updates ?? "unknown"} recorded flow updates. These
        counts do not establish representation quality.
      </p>
    </details>
  );
}
