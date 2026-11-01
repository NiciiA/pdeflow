export const metricNames: Record<string, string> = {
  mse: "MSE",
  rmse: "RMSE",
  mae: "MAE",
  relative_l2: "Relative L2",
  weighted_mse: "Weighted MSE",
  region_balanced_mse: "Region-balanced MSE",
};
export const metricHelp: Record<string, string> = {
  mse: "Mean squared error over the physical field at the current time.",
  rmse: "Square root of whole-field MSE, in field units.",
  mae: "Mean absolute error over the physical field.",
  relative_l2:
    "Error L2 norm divided by reference L2 norm; unavailable near zero reference RMS (≤1e-10).",
  weighted_mse:
    "Squared error weighted by detached reference contrast, normalized by total weight.",
  region_balanced_mse:
    "Blends whole-field MSE with equal means over detected hot/cold regions. Experimental; not source tracking.",
};
export type FieldMetricReport = {
  [key: string]: unknown;
  mse: number;
  rmse: number;
  mae: number;
  relative_l2: number | null;
  weighted_mse: number;
  region_balanced_mse: number | null;
  labels: number[][];
  region_contribution: number[][] | null;
  signed_error: number[][];
  squared_error: number[][];
  weighted_contribution: number[][];
  regions: {
    label: number;
    sign: string;
    area_fraction: number;
    mse: number | null;
  }[];
  region_count: number;
  time: number;
  reference_id: string;
  settings: Record<string, number | string>;
  background_mse: number | null;
};
