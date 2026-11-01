import { useState } from "react";
import { api } from "./types";
import { HeatCanvas } from "./HeatCanvas";
import { bounds } from "./heat";
const names: Record<string, string> = {
  mse: "Standard MSE — default",
  weighted_mse: "Weighted MSE — emphasizes strong features",
  region_balanced_mse: "Region-balanced MSE — experimental",
  mae: "MAE — absolute error",
  spectral_mse: "MSE + phase-sensitive spectral error — experimental",
};
export function FieldLossControls({
  value,
  onChange,
  request,
  queryPoints,
}: {
  value: Record<string, unknown>;
  onChange: (value: Record<string, unknown>) => void;
  request: unknown;
  queryPoints: number;
}) {
  const [preview, setPreview] = useState<{
    examples: {
      id: string;
      family: string;
      reference: number[][];
      labels: number[][];
      weights: number[][];
      region_count: number;
    }[];
    meaning: string;
  } | null>(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const type = String(value.type || "mse");
  function set(key: string, v: unknown) {
    setPreview(null);
    onChange({ ...value, [key]: v });
  }
  const numeric = (
    label: string,
    key: string,
    min: number,
    max: number,
    step: number,
  ) => (
    <label key={key}>
      {label}
      <input
        aria-label={label}
        type="number"
        min={min}
        max={max}
        step={step}
        value={Number(
          value[key] ??
            (
              {
                alpha: 0.5,
                contrast_strength: 2,
                relative_threshold: 0.25,
                absolute_threshold: 0.01,
                minimum_area_fraction: 0.003,
                smoothing_fraction: 0,
                analysis_resolution: 32,
                time_groups: 4,
                region_queries: 256,
                spectral_weight: 0.1,
                spectral_resolution: 16,
                spectral_frequency_power: 0,
              } as Record<string, number>
            )[key],
        )}
        onChange={(e) => set(key, e.target.valueAsNumber)}
      />
    </label>
  );
  return (
    <div className="field-loss-controls">
      <label>
        Field reconstruction loss
        <select
          aria-label="Field reconstruction loss"
          value={type}
          onChange={(e) => {
            setPreview(null);
            onChange({
              ...value,
              type: e.target.value,
              ...(e.target.value === "spectral_mse"
                ? { spectral_weight: 0.1 }
                : value.spectral_weight !== undefined
                  ? { spectral_weight: 0 }
                  : {}),
            });
          }}
        >
          {Object.entries(names).map(([key, name]) => (
            <option key={key} value={key}>
              {name}
            </option>
          ))}
        </select>
      </label>
      {type === "region_balanced_mse" && (
        <>
          {numeric("Feature balance: α", "alpha", 0, 1, 0.05)}
          <p>
            Adds equal emphasis across detected features. Experimental; may
            trade whole-field accuracy for feature accuracy. α=0.5 is a starting
            value, not a validated optimum.
          </p>
        </>
      )}
      {type === "weighted_mse" &&
        numeric("Contrast emphasis: λ", "contrast_strength", 0, 100, 0.1)}
      {type === "spectral_mse" && (
        <>
          {numeric("Spectral weight", "spectral_weight", 0.001, 100, 0.01)}
          {numeric(
            "Structured spectral grid per axis",
            "spectral_resolution",
            8,
            64,
            1,
          )}
          {numeric(
            "Spectral frequency power",
            "spectral_frequency_power",
            0,
            4,
            0.1,
          )}
          <p>
            Extra structured periodic-grid queries per quantity:{" "}
            {Number(value.spectral_resolution ?? 16) ** 2}. Complex coefficient
            differences preserve phase. Separate from region balancing and
            observation likelihood.
          </p>
        </>
      )}
      <small>
        A1 trains the encoder/decoder with this field loss. A2 uses the same
        decoded field loss through learned evolution, with the representation
        frozen, plus transition/composition losses. B keeps its latent
        flow-matching velocity loss, with an independently configured optional
        endpoint observation likelihood. {queryPoints} whole-domain queries per
        supervised quantity/time
        {` + ${type === "region_balanced_mse" ? (value.region_queries ?? 256) : 0} additional region-training queries per supervised quantity/time`}
        . Existing runs resume their recorded objective.
      </small>
      <p>
        Training loss controls learning. Display metrics only change evaluation.
      </p>
      {type === "region_balanced_mse" && (
        <>
          <details>
            <summary>Advanced region analysis</summary>
            <div className="training-form-grid">
              {numeric(
                "Analysis grid per axis",
                "analysis_resolution",
                8,
                128,
                1,
              )}
              {numeric("Supervised time groups", "time_groups", 1, 16, 1)}
              {type === "region_balanced_mse" && (
                <>
                  {numeric(
                    "Relative contrast threshold",
                    "relative_threshold",
                    0.001,
                    1,
                    0.01,
                  )}
                  {numeric(
                    "Absolute contrast floor (training scale)",
                    "absolute_threshold",
                    0,
                    100,
                    0.001,
                  )}
                  {numeric(
                    "Minimum region area fraction",
                    "minimum_area_fraction",
                    0.0001,
                    0.5,
                    0.001,
                  )}
                  {numeric(
                    "Detection smoothing / shortest domain",
                    "smoothing_fraction",
                    0,
                    0.25,
                    0.01,
                  )}
                  {numeric(
                    "Additional region queries / solution",
                    "region_queries",
                    1,
                    8192,
                    1,
                  )}
                </>
              )}
            </div>
            <p>
              Periodic four-neighbour regions are detected from references at
              each supervised physical time. Small and large regions have equal
              region-term importance; merged regions count as one. Background
              remains in the whole-field term. Regions may influence whole-field
              accuracy.
            </p>
          </details>
          <details>
            <summary>Preview regions on training examples</summary>
            <button
              type="button"
              disabled={loading}
              onClick={async () => {
                setLoading(true);
                setError("");
                try {
                  setPreview(await api("/training/loss-preview", request));
                } catch (e) {
                  setError(String(e));
                } finally {
                  setLoading(false);
                }
              }}
            >
              {loading
                ? "Preparing bounded preview…"
                : "Preview training regions"}
            </button>
            {error && <p role="alert">{error}</p>}
            {preview && (
              <>
                <p>{preview.meaning}</p>
                <div className="loss-preview-grid">
                  {preview.examples.map((e) => (
                    <div key={e.id}>
                      <HeatCanvas
                        values={e.reference}
                        range={bounds([e.reference])}
                        label={`${e.family}: ${e.region_count} regions`}
                        regions={e.labels}
                      />
                      <small>
                        {e.region_count} regions · effective objective weight{" "}
                        {Math.min(...e.weights.flat()).toPrecision(2)}–
                        {Math.max(...e.weights.flat()).toPrecision(3)}
                      </small>
                    </div>
                  ))}
                </div>
              </>
            )}
          </details>
        </>
      )}
    </div>
  );
}
