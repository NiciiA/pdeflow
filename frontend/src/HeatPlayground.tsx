import { ReconstructionDiagnostics } from "./ReconstructionDiagnostics";
import { useEffect, useMemo, useRef, useState } from "react";
import {
  Play,
  Pause,
  RotateCcw,
  Thermometer,
  ArrowRight,
  ArrowLeft,
  SlidersHorizontal,
  Database,
  GraduationCap,
  Upload,
  Check,
  Shuffle,
} from "lucide-react";
import {
  call,
  defaults,
  paint,
  bounds,
  download,
  interpolateFrames,
  physicalPoint,
  samplingTimes,
} from "./heat";
import type {
  HeatConfig,
  Reference,
  Point,
  HeatModel,
  Reconstruction,
} from "./heat";
import { HeatCanvas, ColorBar } from "./HeatCanvas";
import { spatialMarginals, marginalScales } from "./heatProfiles";
import type { ProfileMode } from "./heatProfiles";
import { metricNames, metricHelp } from "./fieldMetrics";
import type { FieldMetricReport } from "./fieldMetrics";
import { HeatStorage } from "./HeatStorage";
import { TrainModel } from "./TrainModel";
import "./heat.css";
function Numeric({
  label,
  value,
  onChange,
  min,
  max,
  step = 0.01,
  disabled = false,
}: {
  label: string;
  value: number;
  onChange: (v: number) => void;
  min: number;
  max: number;
  step?: number;
  disabled?: boolean;
}) {
  return (
    <label>
      {label}
      <input
        type="number"
        disabled={disabled}
        aria-label={label}
        min={min}
        max={max}
        step={step}
        value={Number.isFinite(value) ? value : ""}
        onChange={(e) => onChange(e.target.valueAsNumber)}
      />
    </label>
  );
}

const presets = [
  ["cosine", "Cosine waves"],
  ["spot", "Hot spot"],
  ["two", "Two hot spots"],
  ["random", "Smooth random"],
  ["draw", "Draw your own"],
  ["formula", "Formula"],
];
const wavePresets = [
  ["standing", "Standing mode"],
  ["travelling", "Travelling wave"],
  ["pulse", "Displacement pulse"],
  ["velocity_pulse", "Velocity-only pulse"],
  ["signed_pulses", "Signed pulses"],
  ["random", "Random u and v"],
  ["draw", "Draw your own"],
];
const initial = Array.from({ length: 64 }, (_, x) =>
  Array.from(
    { length: 64 },
    (_, y) =>
      0.2 +
      0.8 * Math.cos((2 * Math.PI * x) / 64) * Math.cos((2 * Math.PI * y) / 64),
  ),
);
function loadConfig() {
  try {
    return {
      ...defaults,
      ...JSON.parse(localStorage.getItem("pdeflow.heat.preferences") || "{}"),
      initial: null,
      velocity_initial: null,
    };
  } catch {
    return defaults;
  }
}
type Job = {
  id: string;
  status: string;
  progress: number;
  stage: string;
  error?: { message: string };
};
export default function HeatPlayground() {
  const [cfg, setCfg] = useState<HeatConfig>(loadConfig),
    [reference, setReference] = useState<Reference | null>(null),
    [pending, setPending] = useState(true),
    [referenceError, setReferenceError] = useState("");
  const [page, setPage] = useState("play"),
    [playing, setPlaying] = useState(false),
    [index, setIndex] = useState(0),
    [speed, setSpeed] = useState(1),
    [view, setView] = useState("numerical");
  const [quantity, setQuantity] = useState<"u" | "v">("u"),
    [drawTarget, setDrawTarget] = useState<"u" | "v">("u"),
    [measurementQuantity, setMeasurementQuantity] = useState<"u" | "v">("u");
  const [showProfiles, setShowProfiles] = useState(true),
    [profileMode, setProfileMode] = useState<ProfileMode>("signed");
  const [count, setCount] = useState(50),
    [noise, setNoise] = useState(0),
    [sampleSeed, setSampleSeed] = useState(91),
    [rows, setRows] = useState<Point[]>([]),
    [sampling, setSampling] = useState(false),
    [sampleRecord, setSampleRecord] = useState<unknown>(null),
    [pattern, setPattern] = useState("current"),
    [manual, setManual] = useState<number[][]>([]),
    [placement, setPlacement] = useState("random"),
    [backgroundFraction, setBackgroundFraction] = useState(0.25),
    [focusSpread, setFocusSpread] = useState(0.08),
    [candidates, setCandidates] = useState<number[][]>([]),
    [candidateIdentity, setCandidateIdentity] = useState(""),
    [placementMessage, setPlacementMessage] = useState(""),
    [selectedTimes, setSelectedTimes] = useState("0, 0.5, 1"),
    [adaptive, setAdaptive] = useState(false),
    [manualX, setManualX] = useState(0.5),
    [manualY, setManualY] = useState(0.5),
    [manualValue, setManualValue] = useState(0),
    [placing, setPlacing] = useState(false),
    [pointsOnly, setPointsOnly] = useState(false);
  const [inferDiffusivity, setInferDiffusivity] = useState(false);
  const [models, setModels] = useState<HeatModel[]>([]),
    [model, setModel] = useState(""),
    [modelsLoading, setModelsLoading] = useState(true),
    [experimental, setExperimental] = useState(false),
    [revision, setRevision] = useState(0),
    [device, setDevice] = useState("cpu"),
    [steps, setSteps] = useState(8),
    [particles, setParticles] = useState(2),
    [detail, setDetail] = useState(32),
    [result, setResult] = useState<Reconstruction | null>(null),
    [job, setJob] = useState<Job | null>(null),
    [submitting, setSubmitting] = useState(false),
    [error, setError] = useState("");
  const [errorMetric, setErrorMetric] = useState("rmse"),
    [errorMap, setErrorMap] = useState("absolute"),
    [showRegions, setShowRegions] = useState(false);
  const [contributionLimits, setContributionLimits] = useState({
    job: "",
    weighted: 0,
    region: 0,
  });
  const [fieldMetrics, setFieldMetrics] = useState<FieldMetricReport | null>(
      null,
    ),
    [metricError, setMetricError] = useState("");
  const [brush, setBrush] = useState("add"),
    [radius, setRadius] = useState(0.08),
    [strength, setStrength] = useState(0.2),
    [draft, setDraft] = useState<number[][] | null>(null),
    [undo, setUndo] = useState<number[][][]>([]);
  const epoch = useRef(0),
    sampleEpoch = useRef(0),
    sampleController = useRef<AbortController | null>(null),
    activeSampling = useRef(0),
    inverseEpoch = useRef(0),
    inFlight = useRef(false),
    drawRef = useRef<number[][] | null>(null),
    drawStart = useRef<number[][] | null>(null),
    resultEpoch = useRef(0),
    importInput = useRef<HTMLInputElement>(null);
  function invalidateResult() {
    inverseEpoch.current++;
    setResult(null);
    setError("");
  }
  function invalidateRows() {
    sampleEpoch.current++;
    setRows([]);
    setSampleRecord(null);
    invalidateResult();
  }
  function edit(patch: Partial<HeatConfig>) {
    if (JSON.stringify({ ...cfg, ...patch }) === JSON.stringify(cfg)) return;
    epoch.current++;
    invalidateRows();
    setPlaying(false);
    setIndex(0);
    setCfg((c) => ({ ...c, ...patch }));
    setPending(true);
    setReferenceError("");
  }
  const key = JSON.stringify(cfg);
  useEffect(() => {
    const controller = new AbortController();
    const current = epoch.current;
    const timer = setTimeout(() => {
      void call<Reference>("/playground/reference", cfg, controller.signal)
        .then((r) => {
          if (current !== epoch.current) return;
          setReference(r);
          setPending(false);
          setReferenceError("");
          setIndex((i) => Math.min(i, cfg.end));
        })
        .catch((e) => {
          if (!controller.signal.aborted && current === epoch.current) {
            setReferenceError(e.message);
            setPending(false);
          }
        });
    }, 120);
    try {
      localStorage.setItem(
        "pdeflow.heat.preferences",
        JSON.stringify({
          ...cfg,
          initial: null,
          velocity_initial: null,
          preset: cfg.preset === "draw" ? "cosine" : cfg.preset,
        }),
      );
    } catch {
      /* Preferences are optional. */
    }
    return () => {
      clearTimeout(timer);
      controller.abort();
    };
    // The serialized experiment is the dependency; edits invalidate requests synchronously.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key]);
  const problem = useMemo(
    () =>
      reference
        ? {
            ...reference.problem,
            known_parameters: inferDiffusivity
              ? {}
              : reference.problem.known_parameters,
            unknown_parameters: inferDiffusivity
              ? [
                  reference.problem.family === "wave"
                    ? "wave_speed"
                    : "diffusivity",
                ]
              : [],
            grid: { ...reference.problem.grid, shape: [detail, detail] },
          }
        : null,
    [reference, detail, inferDiffusivity],
  );
  useEffect(() => {
    if (!problem) return;
    let live = true;
    void call<HeatModel[]>("/playground/models", { problem })
      .then((ms) => {
        if (!live) return;
        setModels(ms);
        setModel((old) =>
          ms.some((m) => m.id === old)
            ? old
            : ms.find((m) => m.compatible)?.id || ms[0]?.id || "",
        );
        setModelsLoading(false);
      })
      .catch((e) => {
        if (live) {
          setError(e.message);
          setModelsLoading(false);
        }
      });
    return () => {
      live = false;
    };
  }, [problem, revision]);
  const selected = models.find((m) => m.id === model);
  const physicalTime = index;
  const referenceFrames = useMemo(
    () =>
      (quantity === "v" ? reference?.velocity_frames : reference?.frames) || [
        initial.map((r) => r.map(() => 0)),
      ],
    [quantity, reference],
  );
  const resultFrames = useMemo(
    () => (quantity === "v" ? result?.velocity_frames : result?.frames) || [],
    [quantity, result],
  );
  const parameterName = cfg.family === "wave" ? "wave_speed" : "diffusivity";
  const scalarLabel = cfg.family === "wave" ? "Wave speed c" : "Diffusivity D";
  const [evaluated, setEvaluated] = useState<{
    reference: string;
    quantity: string;
    job: string;
    time: number;
    frame: number[][] | null;
    exact: number[][] | null;
    model: number[][] | null;
  } | null>(null);
  const [evaluating, setEvaluating] = useState(false);
  const candidateKey = JSON.stringify([
    reference?.id,
    measurementQuantity,
    count,
    sampleSeed,
    physicalTime,
    pattern,
    placement,
    backgroundFraction,
    focusSpread,
    selectedTimes,
    adaptive,
  ]);
  useEffect(() => {
    const controller = new AbortController();
    sampleEpoch.current++;
    if (!reference || playing || pending) return;
    const timer = setTimeout(() => {
      setEvaluating(true);
      const refPromise = pointsOnly
        ? Promise.resolve(null)
        : call<{ frame: number[][]; exact: number[][] | null }>(
            `/playground/reference/${reference.id}/at?time=${physicalTime}&quantity=${quantity}`,
            undefined,
            controller.signal,
          );
      const modelPromise =
        result && job?.status === "succeeded"
          ? call<{ frame: number[][] }>(
              `/playground/results/${job.id}/at?time=${physicalTime}&quantity=${quantity}`,
              undefined,
              controller.signal,
            )
          : Promise.resolve(null);
      void Promise.all([refPromise, modelPromise])
        .then(([r, m]) => {
          if (controller.signal.aborted) return;
          setEvaluated({
            reference: reference.id,
            quantity,
            job: job?.id || "",
            time: physicalTime,
            frame: r?.frame || null,
            exact: r?.exact || null,
            model: m?.frame || null,
          });
          setEvaluating(false);
        })
        .catch((e) => {
          if (!controller.signal.aborted) {
            setError(e.message);
            setEvaluating(false);
          }
        });
    }, 100);
    return () => {
      clearTimeout(timer);
      controller.abort();
    };
  }, [
    reference,
    result,
    job?.id,
    job?.status,
    physicalTime,
    quantity,
    playing,
    pending,
    pointsOnly,
  ]);
  useEffect(() => {
    const controller = new AbortController();
    if (!reference || pending || pointsOnly || placement === "manual") return;
    const timer = setTimeout(() => {
      void call<{
        positions: number[][];
        placement: { message: string; reference_assisted: boolean };
      }>(
        "/playground/candidates",
        {
          reference_id: reference.id,
          quantity: measurementQuantity,
          count,
          noise: 0,
          seed: sampleSeed,
          time: physicalTime,
          pattern,
          placement,
          background: backgroundFraction,
          spread: focusSpread,
          times: pattern === "current" ? [] : samplingTimes(selectedTimes),
          adaptive,
        },
        controller.signal,
      )
        .then((r) => {
          if (!controller.signal.aborted) {
            setCandidateIdentity(candidateKey);
            setCandidates(r.positions);
            setPlacementMessage(
              r.placement.message ||
                (r.placement.reference_assisted
                  ? "Reference-assisted: positions use hot and cold features."
                  : ""),
            );
          }
        })
        .catch((e) => {
          if (!controller.signal.aborted) {
            setCandidateIdentity(candidateKey);
            setCandidates([]);
            setPlacementMessage(e.message);
          }
        });
    }, 140);
    return () => {
      clearTimeout(timer);
      controller.abort();
    };
  }, [
    reference,
    pending,
    pointsOnly,
    candidateKey,
    measurementQuantity,
    count,
    sampleSeed,
    physicalTime,
    pattern,
    placement,
    backgroundFraction,
    focusSpread,
    manual,
    selectedTimes,
    adaptive,
  ]);
  useEffect(() => {
    sampleEpoch.current++;
    sampleController.current?.abort();
  }, [
    physicalTime,
    placement,
    backgroundFraction,
    focusSpread,
    manual,
    selectedTimes,
    adaptive,
    pattern,
  ]);
  useEffect(() => {
    if (!playing) return;
    const timer = setInterval(
      () =>
        setIndex((i) => {
          if (i >= cfg.end) {
            setPlaying(false);
            return i;
          }
          return Math.min(cfg.end, i + cfg.end / (cfg.frame_count - 1));
        }),
      100 / speed,
    );
    return () => clearInterval(timer);
  }, [playing, speed, cfg.end, cfg.frame_count]);
  useEffect(() => {
    if (!job || !["queued", "running"].includes(job.status)) return;
    let live = true,
      fetching = false;
    const refresh = async () => {
      if (fetching) return;
      fetching = true;
      try {
        const j = await call<Job>(`/jobs/${job.id}`);
        if (!live) return;
        if (
          j.status === "succeeded" &&
          resultEpoch.current === inverseEpoch.current
        ) {
          if (!pointsOnly && reference) {
            try {
              await call(`/coordinate/jobs/${j.id}/compare`, {
                reference_id: reference.id,
              });
            } catch (e) {
              if (live)
                setError(`Comparison unavailable: ${(e as Error).message}`);
            }
          }
          const r = await call<Reconstruction>(`/playground/results/${j.id}`);
          if (live && resultEpoch.current === inverseEpoch.current) {
            setResult(r);
            setPlaying(false);
          }
        }
        if (live) setJob(j);
        if (j.status === "failed")
          setError(
            j.error?.message ||
              "Reconstruction failed. Check the model and retry.",
          );
      } catch (e) {
        if (live) setError((e as Error).message);
      } finally {
        fetching = false;
      }
    };
    void refresh();
    const timer = setInterval(() => void refresh(), 600);
    return () => {
      live = false;
      clearInterval(timer);
    };
  }, [job, pointsOnly, reference]);
  const running =
    submitting || (!!job && ["queued", "running"].includes(job.status));
  const editable = cfg.preset === "draw" && !placing;
  const refFrames = referenceFrames;
  const frame =
    evaluated?.quantity === quantity &&
    evaluated?.reference === reference?.id &&
    evaluated?.time === physicalTime &&
    evaluated.frame &&
    !playing
      ? evaluated.frame
      : interpolateFrames(
          refFrames,
          reference?.problem.grid.times || [0],
          physicalTime,
        );
  const exact =
    evaluated?.quantity === quantity &&
    evaluated?.reference === reference?.id &&
    evaluated?.time === physicalTime &&
    !playing
      ? evaluated?.exact
      : quantity === "u" && reference?.exact
        ? interpolateFrames(
            reference.exact,
            reference.problem.grid.times,
            physicalTime,
          )
        : null;
  const shown =
    draft ||
    (view === "exact" && exact
      ? exact
      : view === "difference" && exact
        ? frame.map((r, x) => r.map((v, y) => Math.abs(v - exact[x][y])))
        : frame);
  const rawDisplayRange =
    view === "difference" && !draft
      ? bounds([shown])
      : (quantity === "v" ? reference?.velocity_range : reference?.range) || [
          -0.6, 1,
        ];
  const displayRange =
    cfg.family === "wave" && view !== "difference"
      ? [
          -Math.max(...rawDisplayRange.map(Math.abs)),
          Math.max(...rawDisplayRange.map(Math.abs)),
        ]
      : rawDisplayRange;
  const rawComparisonRange = useMemo(
    () =>
      result && reference
        ? bounds([...referenceFrames, ...resultFrames])
        : result
          ? bounds(resultFrames)
          : [0, 1],
    [result, reference, referenceFrames, resultFrames],
  );
  const comparisonRange =
    cfg.family === "wave"
      ? [
          -Math.max(...rawComparisonRange.map(Math.abs)),
          Math.max(...rawComparisonRange.map(Math.abs)),
        ]
      : rawComparisonRange;
  const modelFrame = useMemo(
    () =>
      result
        ? evaluated?.quantity === quantity &&
          evaluated?.job === job?.id &&
          evaluated?.time === physicalTime &&
          evaluated.model &&
          !playing
          ? evaluated.model
          : interpolateFrames(
              resultFrames,
              result.problem.grid.times,
              physicalTime,
            )
        : null,
    [result, evaluated, job?.id, physicalTime, playing, quantity, resultFrames],
  );
  // Query grids are chosen to divide the 64-node reference, so comparisons use actual reference nodes.
  const comparedReference =
    modelFrame && reference
      ? modelFrame.map((r, x) =>
          r.map(
            (_, y) =>
              frame[(x * 64) / modelFrame.length][
                (y * 64) / modelFrame[0].length
              ],
          ),
        )
      : null;
  const difference =
    modelFrame && comparedReference
      ? modelFrame.map((r, x) =>
          r.map((v, y) => Math.abs(v - comparedReference[x][y])),
        )
      : null;
  const errorRange = useMemo(() => {
    let max = 0;
    if (result && reference && !pointsOnly)
      resultFrames.forEach((f, t) =>
        f.forEach((r, x) =>
          r.forEach((v, y) => {
            max = Math.max(
              max,
              Math.abs(
                v -
                  interpolateFrames(
                    referenceFrames,
                    reference.problem.grid.times,
                    result.problem.grid.times[t],
                  )[(x * 64) / f.length][(y * 64) / r.length],
              ),
            );
          }),
        ),
      );
    return [0, max];
  }, [result, reference, pointsOnly, referenceFrames, resultFrames]);
  // No extra scientific calls: bound scales over existing playback arrays.
  const plotLengths =
    reference?.problem.grid.domain.lengths || defaults.lengths;
  const plotOrigins =
    reference?.problem.grid.domain.origins || defaults.origins;
  const cachedProfiles = useMemo(
    () =>
      [...referenceFrames, ...resultFrames].map((f) =>
        spatialMarginals(f, plotLengths, profileMode),
      ),
    [referenceFrames, resultFrames, plotLengths, profileMode],
  );
  const profileScales = marginalScales(
    [
      ...cachedProfiles,
      spatialMarginals(shown, plotLengths, profileMode),
      ...(modelFrame
        ? [spatialMarginals(modelFrame, plotLengths, profileMode)]
        : []),
    ],
    Math.max(...(result ? comparisonRange : displayRange).map(Math.abs)) *
      Math.max(...plotLengths) *
      1e-12,
  );
  useEffect(() => {
    if (
      !modelFrame ||
      !reference ||
      pointsOnly ||
      playing ||
      evaluated?.job !== job?.id ||
      evaluated?.reference !== reference.id ||
      evaluated?.time !== physicalTime ||
      !evaluated?.model
    )
      return;
    const controller = new AbortController();
    const timer = setTimeout(() => {
      call<FieldMetricReport>(
        "/playground/error-metrics",
        {
          reference_id: reference.id,
          quantity,
          time: physicalTime,
          prediction: modelFrame,
          field_loss: result?.metrics.field_loss_provenance || {},
          normalization_scale:
            result?.metrics.normalizations?.[quantity === "u" ? 0 : 1]?.scale ||
            result?.metrics.field_normalization_scale ||
            1,
        },
        controller.signal,
      )
        .then((v) => {
          if (controller.signal.aborted) return;
          const maximum = (v: number[][] | null) =>
            v ? Math.max(0, ...v.flat()) : 0;
          setContributionLimits((old) => ({
            job: job?.id || "",
            weighted: Math.max(
              old.job === job?.id ? old.weighted : 0,
              maximum(v.weighted_contribution),
            ),
            region: Math.max(
              old.job === job?.id ? old.region : 0,
              maximum(v.region_contribution),
            ),
          }));
          setFieldMetrics({ ...v, job_id: job?.id });
          setMetricError("");
        })
        .catch((e) => {
          if (!controller.signal.aborted) setMetricError(String(e));
        });
    }, 120);
    return () => {
      clearTimeout(timer);
      controller.abort();
    };
  }, [
    modelFrame,
    quantity,
    reference,
    physicalTime,
    result,
    pointsOnly,
    playing,
    job?.id,
    evaluated,
  ]);
  const currentMetrics =
    !playing &&
    fieldMetrics?.time === physicalTime &&
    fieldMetrics?.reference_id === reference?.id &&
    fieldMetrics?.job_id === job?.id
      ? fieldMetrics
      : null;
  const mapNames: Record<string, string> = {
    absolute: "Absolute error |model − reference|",
    signed: "Signed error model − reference",
    squared: "Squared error (model − reference)²",
    weighted: "Weighted squared-error contribution",
    region: "Region-balanced MSE contribution",
  };
  const mappedError =
    errorMap === "absolute"
      ? difference
      : errorMap === "signed"
        ? currentMetrics?.signed_error
        : errorMap === "squared"
          ? currentMetrics?.squared_error
          : errorMap === "weighted"
            ? currentMetrics?.weighted_contribution
            : currentMetrics?.region_contribution;
  const errorLimit = errorRange[1];
  const mapRange =
    errorMap === "absolute"
      ? errorRange
      : errorMap === "signed"
        ? [-errorLimit, errorLimit]
        : errorMap === "squared"
          ? [0, errorLimit ** 2]
          : [
              0,
              contributionLimits.job === job?.id
                ? errorMap === "weighted"
                  ? contributionLimits.weighted
                  : contributionLimits.region
                : 0,
            ];
  async function sample(resample = false) {
    if (!reference || pending || referenceError) return;
    if (!Number.isInteger(count) || count < 1 || count > 2000) {
      setError("Choose between 1 and 2000 scalar measurements.");
      return;
    }
    if (!Number.isFinite(noise) || noise < 0) {
      setError(
        "Noise must be a finite, nonnegative absolute standard deviation.",
      );
      return;
    }
    const current = ++sampleEpoch.current;
    const requestId = ++activeSampling.current;
    sampleController.current?.abort();
    const controller = new AbortController();
    sampleController.current = controller;
    const seed = sampleSeed + (resample ? 1 : 0);
    const total = placement === "manual" ? manual.length : count;
    if (total < 1) {
      setError("Add at least one sensor point.");
      return;
    }
    if (resample) setSampleSeed(seed);
    setSampling(true);
    setError("");
    invalidateResult();
    try {
      const r = await call<{ rows: Point[] }>(
        "/playground/measurements",
        {
          reference_id: reference.id,
          quantity: measurementQuantity,
          count: total,
          noise,
          seed,
          time: physicalTime,
          pattern,
          positions: manual,
          placement,
          background: backgroundFraction,
          spread: focusSpread,
          times: pattern === "current" ? [] : samplingTimes(selectedTimes),
          adaptive,
        },
        controller.signal,
      );
      if (current !== sampleEpoch.current) return;
      setRows(r.rows);
      setSampleRecord(r);
    } catch (e) {
      if (current === sampleEpoch.current) setError((e as Error).message);
    } finally {
      if (requestId === activeSampling.current) setSampling(false);
    }
  }
  async function reconstruct() {
    if (inFlight.current || running) return;
    setError("");
    if (!rows.length) {
      setError("Take at least one measurement before running this model.");
      return;
    }
    if (!selected?.compatible) {
      setError(
        selected?.reason ||
          "No trained model yet. Train or import a resolution-flexible 2D heat model.",
      );
      return;
    }
    if (selected.pipeline_check && !experimental) {
      setError(
        "This model is barely trained. Enable experimental use to run it.",
      );
      return;
    }
    inFlight.current = true;
    setSubmitting(true);
    const current = inverseEpoch.current;
    resultEpoch.current = current;
    try {
      const j = await call<Job>("/coordinate/reconstruct", {
        model_id: model,
        require_continuous: true,
        problem,
        rows,
        steps,
        particles,
        device,
        seed: 91,
        chunk_size: 1024,
        submission_token: crypto.randomUUID(),
      });
      setJob(j);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      inFlight.current = false;
      setSubmitting(false);
    }
  }
  function draw(x: number, y: number) {
    if (placing) {
      const point = physicalPoint(x, y, cfg.lengths, cfg.origins);
      setManual((old) => {
        const nearby = old.findIndex(
          (p) =>
            Math.hypot(
              (p[0] - point[0]) / cfg.lengths[0],
              (p[1] - point[1]) / cfg.lengths[1],
            ) < 0.025,
        );
        return nearby >= 0
          ? old.filter((_, i) => i !== nearby)
          : [...old, point];
      });
      return;
    }
    if (!drawRef.current) {
      drawStart.current =
        (drawTarget === "v"
          ? reference?.initial_velocity
          : reference?.initial) || initial;
      drawRef.current = drawStart.current;
      setPlaying(false);
      setIndex(0);
    }
    drawRef.current = paint(drawRef.current, x, y, radius, strength, brush);
    setDraft(drawRef.current);
  }
  function finishDrawing() {
    if (placing || !drawRef.current) return;
    setUndo((u) => [...u.slice(-9), drawStart.current!]);
    edit(
      drawTarget === "v"
        ? {
            velocity_initial: drawRef.current,
            initial: reference?.initial || initial,
            preset: "draw",
          }
        : { initial: drawRef.current, preset: "draw" },
    );
    drawRef.current = null;
    drawStart.current = null;
    setDraft(null);
  }
  async function importModel(file: File) {
    setError("");
    try {
      const response = await fetch("/api/playground/import", {
        method: "POST",
        body: file,
      });
      const r = await response.json();
      if (!response.ok)
        throw new Error(
          r.error?.message || "Choose a compatible coordinate flow export.",
        );
      setRevision((v) => v + 1);
    } catch (e) {
      setError((e as Error).message);
    }
  }
  function transport() {
    return (
      <div className="transport">
        <button
          className="primary play-button"
          disabled={pending || !reference || !!referenceError}
          onClick={() => {
            if (index >= cfg.end) setIndex(0);
            setPlaying((v) => !v);
          }}
        >
          {playing ? <Pause size={17} /> : <Play size={17} />}{" "}
          {playing ? "Pause" : "Play"}
        </button>
        <button
          aria-label="Restart simulation"
          title="Restart from the same initial field"
          onClick={() => {
            setIndex(0);
            setPlaying(false);
          }}
        >
          <RotateCcw size={17} />
        </button>
        <input
          aria-label="Physical time"
          type="range"
          min={0}
          max={cfg.end}
          step={0.001}
          value={index}
          onChange={(e) => {
            sampleEpoch.current++;
            setSampling(false);
            setIndex(+e.target.value);
            setPlaying(false);
          }}
        />
        <output data-testid="physical-time">
          t = {physicalTime.toFixed(3)}
        </output>
        <label className="speed">
          <span className="sr-only">Playback speed</span>
          <select
            aria-label="Playback speed"
            value={speed}
            onChange={(e) => setSpeed(+e.target.value)}
          >
            <option value={0.5}>½×</option>
            <option value={1}>1×</option>
            <option value={2}>2×</option>
            <option value={4}>4×</option>
          </select>
        </label>
      </div>
    );
  }
  return (
    <div className="heat-app">
      <header className="heat-header">
        <button className="heat-brand" onClick={() => setPage("play")}>
          <span>
            <Thermometer size={22} />
          </span>
          PDEFlow <small>Heat & wave playground</small>
        </button>
        <nav aria-label="Application">
          <button onClick={() => setPage(page === "train" ? "play" : "train")}>
            <GraduationCap size={17} />
            Train model
          </button>
          <button
            onClick={() => setPage(page === "storage" ? "play" : "storage")}
          >
            <Database size={17} />
            Storage
          </button>
        </nav>
      </header>
      {page !== "play" ? (
        <main className="heat-secondary">
          <button className="back" onClick={() => setPage("play")}>
            <ArrowLeft size={16} />
            Back to playground
          </button>
          {page === "storage" ? (
            <HeatStorage
              currentModel={model}
              onDeleted={(ids) => {
                if (ids.includes(model)) {
                  setModel("");
                  setExperimental(false);
                  invalidateResult();
                }
                setRevision((r) => r + 1);
              }}
            />
          ) : (
            <TrainModel
              focused
              onRefresh={() => setRevision((r) => r + 1)}
              onUse={(selection) => {
                setModel(selection.id);
                setExperimental(false);
                setRevision((r) => r + 1);
                setPage("play");
                invalidateResult();
              }}
            />
          )}
        </main>
      ) : (
        <main className="heat-main">
          <div className="heat-intro">
            <div>
              <p className="eyebrow">A LITTLE HEAT. A LOT TO EXPLORE.</p>
              <h1>Make a field. Watch it flow.</h1>
              <p>Paint, play, then reconstruct from just a few measurements.</p>
            </div>
            <span className="state-pill" role="status">
              {running
                ? "Model running"
                : result
                  ? "Results ready"
                  : rows.length
                    ? "Measurements ready"
                    : pending
                      ? "Updating simulation"
                      : reference
                        ? "Reference ready"
                        : "Preparing field"}
            </span>
          </div>
          <div className="playground-grid">
            <aside className="heat-controls">
              <section>
                <h2>
                  <span>01</span>Create initial state
                </h2>
                <label>
                  Equation
                  <select
                    aria-label="Equation"
                    value={cfg.family}
                    onChange={(e) => {
                      const family = e.target.value as "heat" | "wave";
                      setQuantity("u");
                      setDrawTarget("u");
                      setMeasurementQuantity("u");
                      setUndo([]);
                      edit({
                        family,
                        preset: family === "wave" ? "standing" : "cosine",
                        initial: null,
                        velocity_initial: null,
                        velocity_preset: "zero",
                      });
                    }}
                  >
                    <option value="heat">Heat</option>
                    <option value="wave">Wave</option>
                  </select>
                </label>
                {cfg.family === "wave" && (
                  <>
                    <label>
                      Initial velocity
                      <select
                        aria-label="Initial velocity"
                        disabled={["travelling", "velocity_pulse"].includes(
                          cfg.preset,
                        )}
                        value={cfg.velocity_preset}
                        onChange={(e) =>
                          edit({
                            velocity_preset: e.target.value,
                            velocity_initial: null,
                          })
                        }
                      >
                        <option value="zero">Explicit zero v₀</option>
                        <option value="random">Independent random v₀</option>
                        <option value="pulse">Velocity pulse</option>
                      </select>
                    </label>
                    <p>
                      Wave state is (u, v = ∂u/∂t). Initial velocity is
                      independent of displacement. Travelling mode supplies its
                      matched velocity.
                    </p>
                  </>
                )}
                <div className="preset-grid">
                  {(cfg.family === "wave" ? wavePresets : presets).map(
                    ([id, label]) => (
                      <button
                        aria-pressed={cfg.preset === id}
                        key={id}
                        onClick={() => {
                          setDrawTarget("u");
                          setQuantity("u");
                          setUndo([]);
                          setDraft(null);
                          setPlacing(false);
                          if (placement === "manual") setPlacement("random");
                          edit({
                            preset: id,
                            velocity_preset:
                              cfg.family === "wave" && id === "random"
                                ? "random"
                                : "zero",
                            velocity_initial: null,
                            initial:
                              id === "draw"
                                ? reference?.initial || initial
                                : null,
                          });
                          setView("numerical");
                        }}
                      >
                        {label}
                        {cfg.preset === id && <Check size={12} />}
                      </button>
                    ),
                  )}
                </div>
                {["cosine", "standing", "travelling"].includes(cfg.preset) && (
                  <div className="number-grid">
                    {
                      <Numeric
                        label={"Amplitude"}
                        value={cfg.amplitude}
                        onChange={(v) => edit({ amplitude: v })}
                        min={-10}
                        max={10}
                      />
                    }
                    {
                      <Numeric
                        label={"Background"}
                        value={cfg.background}
                        onChange={(v) => edit({ background: v })}
                        min={-10}
                        max={10}
                      />
                    }
                    {
                      <Numeric
                        label={"Waves in x"}
                        value={cfg.kx}
                        onChange={(v) => edit({ kx: v })}
                        min={0}
                        max={15}
                        step={1}
                      />
                    }
                    {
                      <Numeric
                        label={"Waves in y"}
                        value={cfg.ky}
                        onChange={(v) => edit({ ky: v })}
                        min={0}
                        max={15}
                        step={1}
                      />
                    }
                  </div>
                )}
                {cfg.preset === "random" && (
                  <>
                    <div className="number-grid">
                      {
                        <Numeric
                          label={"Random field seed"}
                          value={cfg.seed}
                          onChange={(v) => edit({ seed: v })}
                          min={0}
                          max={2 ** 32 - 1}
                          step={1}
                        />
                      }
                    </div>
                    <button onClick={() => edit({ seed: cfg.seed + 1 })}>
                      <Shuffle size={14} />
                      New random field
                    </button>
                  </>
                )}
                {cfg.preset === "formula" && (
                  <div className="formula-editor">
                    <label>
                      Starting field u(x,y,0) =
                      <textarea
                        aria-label="Starting field formula"
                        value={cfg.expression}
                        onChange={(e) => edit({ expression: e.target.value })}
                      />
                    </label>
                    <small>
                      x, y in physical coordinates. pi, sin, cos, exp, sqrt,
                      tanh, abs; + − * / **. D is known; t = 0.
                    </small>
                    <output>{cfg.expression}</output>
                  </div>
                )}
                {editable && (
                  <div className="brush-controls">
                    {cfg.family === "wave" && (
                      <label>
                        Draw quantity
                        <select
                          aria-label="Draw quantity"
                          value={drawTarget}
                          onChange={(e) => {
                            const q = e.target.value as "u" | "v";
                            setDrawTarget(q);
                            setQuantity(q);
                            setIndex(0);
                            setUndo([]);
                          }}
                        >
                          <option value="u">Initial displacement u₀</option>
                          <option value="v">Initial velocity v₀</option>
                        </select>
                      </label>
                    )}
                    <p>
                      Pause and edit the starting field, then restart. No live
                      forcing.
                    </p>
                    <label>
                      Brush
                      <select
                        aria-label="Brush"
                        value={brush}
                        onChange={(e) => setBrush(e.target.value)}
                      >
                        <option value="add">Increase field</option>
                        <option value="remove">Decrease field</option>
                        <option value="erase">Erase toward zero</option>
                      </select>
                    </label>
                    <label>
                      Brush size
                      <input
                        aria-label="Brush size"
                        type="range"
                        min={0.02}
                        max={0.3}
                        step={0.01}
                        value={radius}
                        onChange={(e) => setRadius(+e.target.value)}
                      />
                    </label>
                    <label>
                      Strength
                      <input
                        aria-label="Brush strength"
                        type="range"
                        min={0.01}
                        max={1}
                        step={0.01}
                        value={strength}
                        onChange={(e) => setStrength(+e.target.value)}
                      />
                    </label>
                    <div className="heat-actions">
                      <button
                        onClick={() => {
                          setUndo((u) => [
                            ...u,
                            (drawTarget === "v"
                              ? reference?.initial_velocity
                              : reference?.initial) || initial,
                          ]);
                          edit(
                            drawTarget === "v"
                              ? {
                                  velocity_initial: initial.map((r) =>
                                    r.map(() => 0),
                                  ),
                                }
                              : { initial: initial.map((r) => r.map(() => 0)) },
                          );
                        }}
                      >
                        Clear field
                      </button>
                      <button
                        disabled={!undo.length}
                        onClick={() => {
                          edit(
                            drawTarget === "v"
                              ? { velocity_initial: undo.at(-1)! }
                              : { initial: undo.at(-1)! },
                          );
                          setUndo((u) => u.slice(0, -1));
                        }}
                      >
                        Undo stroke
                      </button>
                    </div>
                  </div>
                )}
                {referenceError && (
                  <p role="alert" className="local-error">
                    {referenceError} Your last valid simulation is preserved.
                  </p>
                )}
              </section>
              <section>
                <h2>
                  <span>02</span>Simulate dynamics
                </h2>
                <label className="diffusion-label">
                  {scalarLabel}
                  <input
                    aria-label={scalarLabel}
                    type="number"
                    min={0.0001}
                    max={2}
                    step={0.01}
                    value={
                      Number.isFinite(cfg.family === "wave" ? cfg.c : cfg.D)
                        ? cfg.family === "wave"
                          ? cfg.c
                          : cfg.D
                        : ""
                    }
                    onChange={(e) =>
                      edit(
                        cfg.family === "wave"
                          ? { c: e.target.valueAsNumber }
                          : { D: e.target.valueAsNumber },
                      )
                    }
                  />
                </label>
                <input
                  aria-label={`${scalarLabel} slider`}
                  type="range"
                  min={0.001}
                  max={cfg.family === "wave" ? 0.6 : 0.3}
                  step={0.001}
                  value={cfg.family === "wave" ? cfg.c : cfg.D}
                  onChange={(e) =>
                    edit(
                      cfg.family === "wave"
                        ? { c: +e.target.value }
                        : { D: +e.target.value },
                    )
                  }
                />
                <small>
                  {cfg.family === "wave"
                    ? "Time is dimensionless. c sets wave speed; v is the time derivative of u."
                    : "Time is dimensionless. D controls how quickly heat spreads."}
                </small>
                <p>Time interval: 0 to {cfg.end}</p>
                {
                  <Numeric
                    label={"End time"}
                    value={cfg.end}
                    onChange={(v) => edit({ end: v })}
                    min={0.001}
                    max={5}
                    step={0.01}
                  />
                }
              </section>
              <section>
                <h2>
                  <span>03</span>Take measurements
                </h2>
                {cfg.family === "wave" && (
                  <label>
                    Measurement quantity
                    <select
                      aria-label="Measurement quantity"
                      value={measurementQuantity}
                      onChange={(e) => {
                        setMeasurementQuantity(e.target.value as "u" | "v");
                        setRows([]);
                        invalidateResult();
                      }}
                    >
                      <option value="u">Displacement u</option>
                      <option value="v">Velocity v = ∂u/∂t</option>
                    </select>
                  </label>
                )}
                <div
                  className="placement-choices"
                  role="group"
                  aria-label="Placement"
                >
                  {[
                    ["random", "Random"],
                    ["even", "Evenly spaced"],
                    ["heat", "Feature-focused"],
                    ["manual", "Click points"],
                  ].map(([id, label]) => (
                    <button
                      key={id}
                      aria-pressed={placement === id}
                      disabled={pointsOnly && id === "heat"}
                      onClick={() => {
                        sampleEpoch.current++;
                        setSampling(false);
                        invalidateResult();
                        if (id === "manual" || pattern === "manual")
                          setPattern("current");
                        if (id === "manual") setAdaptive(false);
                        setPlacement(id);
                        setPlacing(id === "manual");
                        setPlaying(false);
                      }}
                    >
                      {label}
                    </button>
                  ))}
                </div>
                <small>
                  {placement === "random"
                    ? "Spread points randomly across the field."
                    : placement === "even"
                      ? "Cover the field with regularly spaced points (approximately even for non-square counts)."
                      : placement === "heat"
                        ? "Place more points near strong heat features while keeping some background coverage."
                        : "Choose measurement locations directly. Drawing is paused while placing sensors."}
                </small>
                <p>
                  Sampling time: <b>{physicalTime.toFixed(3)}</b>
                </p>
                {!pending &&
                  !pointsOnly &&
                  candidateIdentity === candidateKey &&
                  placement !== "manual" &&
                  placementMessage && (
                    <p className="measurement-status">{placementMessage}</p>
                  )}
                {placement === "heat" && (
                  <label>
                    Focus spread
                    <input
                      aria-label="Focus spread"
                      type="range"
                      min={0.04}
                      max={0.3}
                      step={0.01}
                      value={focusSpread}
                      onChange={(e) => {
                        invalidateResult();
                        setFocusSpread(+e.target.value);
                      }}
                    />
                  </label>
                )}
                {placement === "manual" && (
                  <div className="manual-sensors">
                    <p>{manual.length} sensor points · click/tap to toggle</p>
                    <button onClick={() => setManual([])}>
                      Clear sensor points
                    </button>
                    <details>
                      <summary>Enter or remove points</summary>
                      <Numeric
                        label="Sensor x"
                        value={manualX}
                        onChange={setManualX}
                        min={cfg.origins[0]}
                        max={cfg.origins[0] + cfg.lengths[0]}
                      />
                      <Numeric
                        label="Sensor y"
                        value={manualY}
                        onChange={setManualY}
                        min={cfg.origins[1]}
                        max={cfg.origins[1] + cfg.lengths[1]}
                      />
                      <button
                        onClick={() =>
                          setManual((p) => [...p, [manualX, manualY]])
                        }
                      >
                        Add sensor point
                      </button>
                      {manual.map((p, i) => (
                        <button
                          key={i}
                          aria-label={`Remove sensor ${i + 1}`}
                          onClick={() =>
                            setManual((points) =>
                              points.filter((_, j) => i !== j),
                            )
                          }
                        >
                          {i + 1}: {p.map((v) => v.toFixed(3)).join(", ")} ×
                        </button>
                      ))}
                    </details>
                  </div>
                )}
                <div className="number-grid">
                  {
                    <Numeric
                      label={"Number of measurements"}
                      value={placement === "manual" ? manual.length : count}
                      disabled={placement === "manual"}
                      onChange={(v) => {
                        setCount(v);
                        invalidateRows();
                      }}
                      min={1}
                      max={2000}
                      step={1}
                    />
                  }
                  {
                    <Numeric
                      label={"Noise (σ)"}
                      value={noise}
                      onChange={(v) => {
                        setNoise(v);
                        sampleEpoch.current++;
                        setSampling(false);
                      }}
                      min={0}
                      max={100}
                      step={0.01}
                    />
                  }
                </div>
                <div className="heat-actions">
                  <button
                    disabled={
                      pending || !!referenceError || sampling || pointsOnly
                    }
                    onClick={() => void sample()}
                  >
                    {sampling ? "Sampling…" : "Take measurements"}
                  </button>
                  <button
                    aria-label="Resample measurements"
                    disabled={
                      !rows.length ||
                      sampling ||
                      pointsOnly ||
                      placement === "even" ||
                      placement === "manual"
                    }
                    onClick={() => void sample(true)}
                  >
                    <Shuffle size={15} />
                  </button>
                </div>
                {new Set(rows.map((r) => r.time)).size > 1 && (
                  <p>
                    {new Set(rows.map((r) => `${r.x},${r.y}`)).size} spatial
                    sensors · {new Set(rows.map((r) => r.time)).size} sampled
                    times · {rows.length} total scalars
                  </p>
                )}
                {rows.length > 0 && (
                  <p
                    className="measurement-status"
                    data-testid="measurement-summary"
                  >
                    {rows.length} measurements{" "}
                    {new Set(rows.map((r) => r.time)).size === 1
                      ? `at t = ${rows[0].time.toFixed(3)}`
                      : "across physical times"}{" "}
                    · σ = {rows[0].sigma}
                  </p>
                )}
              </section>
            </aside>
            <div
              className="heat-stage"
              data-reference-id={reference?.id}
              data-evaluated-time={
                !playing &&
                evaluated?.quantity === quantity &&
                evaluated?.reference === reference?.id &&
                evaluated?.time === physicalTime
                  ? evaluated.time
                  : "cached display / pending"
              }
            >
              <div className="equation-strip">
                <strong>
                  {cfg.family === "wave"
                    ? "∂²u/∂t² = c² Δu; v = ∂u/∂t"
                    : "∂u/∂t = D(∂²u/∂x² + ∂²u/∂y²)"}
                </strong>
                <span>
                  {cfg.family === "wave"
                    ? "Time is dimensionless. c sets wave speed; v is the time derivative of u."
                    : "Time is dimensionless. D controls how quickly heat spreads."}
                </span>
              </div>
              {cfg.family === "wave" && (
                <label>
                  Display quantity
                  <select
                    aria-label="Display quantity"
                    value={quantity}
                    onChange={(e) => {
                      setQuantity(e.target.value as "u" | "v");
                      setEvaluated(null);
                      setFieldMetrics(null);
                      setContributionLimits({
                        job: "",
                        weighted: 0,
                        region: 0,
                      });
                    }}
                  >
                    <option value="u">Displacement u</option>
                    <option value="v">Velocity v = ∂u/∂t</option>
                  </select>
                </label>
              )}
              <div className="profile-options">
                <label>
                  <input
                    type="checkbox"
                    checked={showProfiles}
                    onChange={(e) => setShowProfiles(e.target.checked)}
                  />{" "}
                  Show profiles
                </label>
                {showProfiles && (
                  <label>
                    <span className="sr-only">Profile display</span>
                    <select
                      aria-label="Profile display"
                      value={profileMode}
                      onChange={(e) =>
                        setProfileMode(e.target.value as ProfileMode)
                      }
                    >
                      <option value="signed">Signed field</option>
                      <option value="magnitude">Magnitude</option>
                    </select>
                  </label>
                )}
              </div>
              {result && modelFrame && !pointsOnly && (
                <div className="error-display-controls">
                  <label title={metricHelp[errorMetric]}>
                    Error metric
                    <select
                      aria-label="Error metric"
                      value={errorMetric}
                      onChange={(e) => setErrorMetric(e.target.value)}
                    >
                      {Object.entries(metricNames).map(([id, name]) => (
                        <option key={id} value={id}>
                          {name}
                        </option>
                      ))}
                    </select>
                  </label>
                  <label>
                    Error map
                    <select
                      aria-label="Error map"
                      value={errorMap}
                      onChange={(e) => setErrorMap(e.target.value)}
                    >
                      {Object.entries(mapNames).map(([id, name]) => (
                        <option key={id} value={id}>
                          {name}
                        </option>
                      ))}
                    </select>
                  </label>
                  <label>
                    <input
                      type="checkbox"
                      checked={showRegions}
                      onChange={(e) => setShowRegions(e.target.checked)}
                    />
                    Show detected regions
                  </label>
                </div>
              )}
              {result && modelFrame ? (
                <>
                  <div
                    className={`comparison-grid ${pointsOnly ? "no-reference" : ""}`}
                    data-testid="comparison"
                  >
                    {!pointsOnly && comparedReference && (
                      <HeatCanvas
                        fieldSymbol={quantity}
                        profileLabel={
                          cfg.family === "wave"
                            ? quantity === "v"
                              ? "Velocity"
                              : "Displacement"
                            : "Heat"
                        }
                        label="Reference"
                        values={comparedReference}
                        profileValues={frame}
                        regions={
                          showRegions ? currentMetrics?.labels : undefined
                        }
                        showProfiles={showProfiles}
                        profileMode={profileMode}
                        profileScales={profileScales}
                        range={comparisonRange}
                        points={rows.filter((r) => r.field === quantity)}
                        time={physicalTime}
                        lengths={plotLengths}
                        origins={plotOrigins}
                      />
                    )}
                    <HeatCanvas
                      fieldSymbol={quantity}
                      profileLabel={
                        cfg.family === "wave"
                          ? quantity === "v"
                            ? "Velocity"
                            : "Displacement"
                          : "Heat"
                      }
                      label="Model reconstruction"
                      values={modelFrame}
                      regions={showRegions ? currentMetrics?.labels : undefined}
                      showProfiles={showProfiles}
                      profileMode={profileMode}
                      profileScales={profileScales}
                      range={comparisonRange}
                      points={rows.filter((r) => r.field === quantity)}
                      time={physicalTime}
                      lengths={plotLengths}
                      origins={plotOrigins}
                    />
                    {!pointsOnly && mappedError && (
                      <HeatCanvas
                        fieldSymbol={quantity}
                        profileLabel={
                          cfg.family === "wave"
                            ? quantity === "v"
                              ? "Velocity"
                              : "Displacement"
                            : "Heat"
                        }
                        label={mapNames[errorMap]}
                        values={mappedError}
                        range={mapRange}
                        regions={
                          showRegions ? currentMetrics?.labels : undefined
                        }
                        points={rows.filter((r) => r.field === quantity)}
                        time={physicalTime}
                        lengths={plotLengths}
                        origins={plotOrigins}
                      />
                    )}
                  </div>
                  <ColorBar
                    range={comparisonRange}
                    label={`Shared ${quantity} scale`}
                  />
                  {!pointsOnly && difference && (
                    <ColorBar range={mapRange} label={mapNames[errorMap]} />
                  )}
                  {metricError && <p role="status">{metricError}</p>}
                  {!pointsOnly && (
                    <details className="error-details">
                      <summary>Error metric details</summary>
                      <p>
                        {metricHelp[errorMetric]} Scalar metrics use identical
                        physical coordinates and time. The independently
                        selected error map is a pointwise discrepancy, not RMSE
                        or relative L2. Display choices never change training or
                        inference.
                      </p>
                      {currentMetrics && (
                        <>
                          <p>
                            Detected regions: {currentMetrics.region_count}. α=
                            {String(currentMetrics.settings.alpha)} · contrast
                            λ=
                            {String(
                              currentMetrics.settings.contrast_strength,
                            )}{" "}
                            · analysis{" "}
                            {String(
                              currentMetrics.settings.analysis_resolution,
                            )}
                            ² · minimum area fraction{" "}
                            {String(
                              currentMetrics.settings.minimum_area_fraction,
                            )}
                            . Reference-derived regions at this physical time,
                            not tracked sources.
                          </p>
                          <pre>
                            {JSON.stringify(
                              {
                                settings: currentMetrics.settings,
                                regions: currentMetrics.regions,
                                background_mse: currentMetrics.background_mse,
                              },
                              null,
                              2,
                            )}
                          </pre>
                        </>
                      )}
                    </details>
                  )}
                  <ReconstructionDiagnostics metrics={result.metrics} />
                  <div className="result-metrics">
                    {!pointsOnly && result.comparison && (
                      <span>
                        <b>
                          {typeof currentMetrics?.[errorMetric] === "number"
                            ? Number(currentMetrics[errorMetric]).toPrecision(4)
                            : "—"}
                        </b>
                        {metricNames[errorMetric]}
                        <small title={metricHelp[errorMetric]}>
                          {playing
                            ? "Pause to evaluate exact physical time"
                            : currentMetrics
                              ? `At physical time ${physicalTime.toFixed(3)} · physical fields`
                              : "Evaluating matched coordinates…"}
                        </small>
                      </span>
                    )}
                    <span>
                      <b>
                        {result.metrics.observation_consistency?.rmse_against_noisy_observed_values_per_sample
                          .map((x) => x.toPrecision(3))
                          .join(" / ") || "—"}
                      </b>
                      Sensor RMSE
                      <small>Per sample vs noisy observed values</small>
                    </span>
                    <span>
                      <b>{result.metrics.seconds?.toFixed(2)} s</b>Inference
                      runtime
                      <small>
                        Integration + decoding; load/export excluded
                      </small>
                    </span>
                  </div>
                </>
              ) : (
                <>
                  <div className="field-topline">
                    <span>
                      {pointsOnly
                        ? "Measurements only · no ground truth"
                        : editable
                          ? "Paint the starting field"
                          : view === "difference"
                            ? "Exact / numerical difference"
                            : "Reference simulation"}
                    </span>
                    {(pending ||
                      (!playing &&
                        (evaluating ||
                          evaluated?.reference !== reference?.id ||
                          evaluated?.time !== physicalTime))) && (
                      <span role="status">
                        {pending
                          ? "Updating…"
                          : "Evaluating requested physical time…"}
                      </span>
                    )}
                    {cfg.preset === "cosine" && !pointsOnly && (
                      <select
                        aria-label="Reference view"
                        value={view}
                        onChange={(e) => setView(e.target.value)}
                      >
                        <option value="numerical">Numerical reference</option>
                        <option value="exact">Exact reference</option>
                        <option value="difference">Difference</option>
                      </select>
                    )}
                  </div>
                  <HeatCanvas
                    fieldSymbol={quantity}
                    profileLabel={
                      cfg.family === "wave"
                        ? quantity === "v"
                          ? "Velocity"
                          : "Displacement"
                        : "Heat"
                    }
                    label={
                      pointsOnly
                        ? "Observation locations (background is not a reference)"
                        : view === "difference"
                          ? "Absolute numerical − exact difference"
                          : `${cfg.family} ${quantity}`
                    }
                    values={
                      pointsOnly ? initial.map((r) => r.map(() => 0)) : shown
                    }
                    range={pointsOnly ? [0, 1] : displayRange}
                    points={rows.filter((r) => r.field === quantity)}
                    time={physicalTime}
                    lengths={plotLengths}
                    origins={plotOrigins}
                    candidates={
                      placement === "manual"
                        ? manual
                        : pending ||
                            pointsOnly ||
                            playing ||
                            candidateIdentity !== candidateKey
                          ? []
                          : candidates
                    }
                    showProfiles={
                      showProfiles && !pointsOnly && view !== "difference"
                    }
                    profileMode={profileMode}
                    profileScales={profileScales}
                    discrete={placing}
                    onPoint={editable || placing ? draw : undefined}
                    onFinish={finishDrawing}
                  />
                  {!pointsOnly && (
                    <ColorBar
                      range={displayRange}
                      label={
                        view === "difference"
                          ? "Absolute difference"
                          : quantity === "u"
                            ? "Field u"
                            : "Velocity v"
                      }
                    />
                  )}
                </>
              )}
              {transport()}
              <div className="stage-footnote">
                <span>↔ Periodic field · opposite edges connect</span>
                <span>
                  {result
                    ? "One fixed generated latent across physical time"
                    : "Reference simulation needs no trained model"}
                </span>
              </div>
              <div className="model-action">
                <div>
                  <h2>Reconstruct from measurements</h2>
                  <label>
                    <input
                      type="checkbox"
                      checked={inferDiffusivity}
                      onChange={(e) => {
                        setInferDiffusivity(e.target.checked);
                        invalidateResult();
                      }}
                    />{" "}
                    Infer{" "}
                    {cfg.family === "wave" ? "wave speed c" : "diffusivity D"}{" "}
                    jointly
                  </label>
                  <p>
                    {inferDiffusivity
                      ? `${scalarLabel} is hidden. The shared model generates a positive scalar with the initial state; inference can be ambiguous.`
                      : `The model receives the specified ${scalarLabel}.`}
                  </p>
                  {result?.metrics.inferred_parameters && (
                    <p>
                      {scalarLabel} samples:{" "}
                      {(result.metrics.inferred_parameters[parameterName] || [])
                        .map((d) => d.toPrecision(4))
                        .join(", ")}
                      . Tiny checkpoints do not establish parameter
                      identification.
                    </p>
                  )}
                  {result?.metrics.physics && (
                    <details>
                      <summary>Physical diagnostics</summary>
                      <p>
                        Measured residuals and grid-based invariant drift; no
                        exact learned conservation guarantee.
                      </p>
                      <pre className="physical-diagnostics">
                        {JSON.stringify(result.metrics.physics, null, 2)}
                      </pre>
                    </details>
                  )}
                  {modelsLoading ? (
                    <p>Looking for compatible models…</p>
                  ) : !models.length ? (
                    <p>No trained model yet. Simulation remains available.</p>
                  ) : (
                    <>
                      <select
                        aria-label="Heat model"
                        value={model}
                        onChange={(e) => {
                          setModel(e.target.value);
                          setExperimental(false);
                          invalidateResult();
                        }}
                      >
                        {models.map((m) => (
                          <option key={m.id} value={m.id}>
                            {m.name}
                            {!m.compatible
                              ? " · unavailable for this experiment"
                              : ""}
                          </option>
                        ))}
                      </select>
                      {selected && !selected.compatible && (
                        <p className="local-error">{selected.reason}</p>
                      )}
                      {selected?.pipeline_check ? (
                        <label className="experimental">
                          <input
                            type="checkbox"
                            checked={experimental}
                            onChange={(e) => setExperimental(e.target.checked)}
                          />
                          Barely trained · allow experimental use (accuracy is
                          poor)
                        </label>
                      ) : (
                        <p>
                          Scientifically unevaluated · inspect held-out
                          validation.
                        </p>
                      )}
                    </>
                  )}
                </div>
                <div className="model-buttons">
                  {!models.length && !modelsLoading ? (
                    <>
                      <button
                        className="primary"
                        onClick={() => setPage("train")}
                      >
                        Train model
                      </button>
                      <button onClick={() => importInput.current?.click()}>
                        <Upload size={14} />
                        Import model
                      </button>
                    </>
                  ) : (
                    <button
                      className="primary"
                      disabled={
                        running || pending || !!referenceError || modelsLoading
                      }
                      onClick={() => void reconstruct()}
                    >
                      {" "}
                      {running
                        ? "Running…"
                        : result
                          ? "Run again"
                          : "Run model"}
                      <ArrowRight size={16} />
                    </button>
                  )}
                </div>
              </div>
              {running && (
                <div className="model-progress" role="status">
                  <progress max={1} value={job?.progress || 0} />
                  <span>
                    {job?.stage?.replaceAll("_", " ") || "Checking model"} ·{" "}
                    {Math.round((job?.progress || 0) * 100)}%
                  </span>
                  {job && (
                    <button
                      onClick={() =>
                        void call(`/jobs/${job.id}/cancel`, {})
                          .then(() => setJob({ ...job, status: "cancelled" }))
                          .catch((e) => setError(e.message))
                      }
                    >
                      Cancel reconstruction
                    </button>
                  )}
                </div>
              )}
              {job?.status === "cancelled" && (
                <p>Reconstruction cancelled. You can retry.</p>
              )}
              {error && (
                <p role="alert" className="local-error">
                  {error}
                </p>
              )}
            </div>
          </div>
          <details className="playground-details">
            <summary>
              <SlidersHorizontal size={15} />
              Advanced & details
            </summary>
            <div className="advanced-grid">
              <section>
                <h3>Domain & reference</h3>
                <p>
                  Periodic boundaries: heat leaving one edge enters through the
                  opposite edge. Periodic constant-D heat and constant-c wave
                  are supported here.
                </p>
                <div className="number-grid">
                  {[0, 1].map((a) => (
                    <div key={a}>
                      {
                        <Numeric
                          label={`Domain length ${a ? "y" : "x"}`}
                          value={cfg.lengths[a]}
                          onChange={(v) =>
                            edit({
                              lengths: cfg.lengths.map((x, i) =>
                                i === a ? v : x,
                              ),
                            })
                          }
                          min={0.05}
                          max={20}
                        />
                      }
                      {
                        <Numeric
                          label={`Domain origin ${a ? "y" : "x"}`}
                          value={cfg.origins[a]}
                          onChange={(v) =>
                            edit({
                              origins: cfg.origins.map((x, i) =>
                                i === a ? v : x,
                              ),
                            })
                          }
                          min={-100}
                          max={100}
                        />
                      }
                    </div>
                  ))}
                  {
                    <Numeric
                      label={"Phase x"}
                      value={cfg.phix}
                      onChange={(v) => edit({ phix: v })}
                      min={-7}
                      max={7}
                    />
                  }
                  {
                    <Numeric
                      label={"Phase y"}
                      value={cfg.phiy}
                      onChange={(v) => edit({ phiy: v })}
                      min={-7}
                      max={7}
                    />
                  }
                </div>
                <p>
                  64 × 64 endpoint-excluded nodes. Numerical reference: finite
                  Fourier interpolant with exponential mode evolution, 31
                  display/export frames (adjustable below). Arbitrary grid
                  fields are not claimed to be exact continuum solutions.
                </p>
                {reference?.exact_max_error != null && (
                  <p data-testid="exact-error">
                    Independent FFT / exact cosine max error:{" "}
                    {reference.exact_max_error.toExponential(3)}.
                  </p>
                )}
                <details>
                  <summary>Display & export frames</summary>
                  <p>
                    Profiles integrate the displayed field across the other
                    spatial direction, with physical cell widths on periodic
                    endpoint-excluded nodes. Signed values can cancel; Magnitude
                    integrates |u| without changing the field. Reference
                    profiles use all 64 × 64 nodes; model profiles use the
                    available query grid. Cached playback interpolates fields
                    before integration. Shared scales cover cached
                    reference/model frames and expand for newly evaluated or
                    drawn values. No additional solver or model calls are made.
                    The error and measurements-only panels omit profiles.
                  </p>
                  <Numeric
                    label="Animation frame count"
                    value={cfg.frame_count}
                    onChange={(v) => {
                      setCfg((c) => ({ ...c, frame_count: v }));
                    }}
                    min={2}
                    max={101}
                    step={1}
                  />
                  <p>
                    Cached playback uses linear display interpolation. Paused
                    scrubbing evaluates the exact requested time using Fourier
                    evolution and the same generated latent. Frame count changes
                    neither training support nor committed observation times.
                  </p>
                </details>
                <details>
                  <summary>Exact cosine solution</summary>
                  <p>
                    u = c + A exp(−D[(2πkx/Lx)²+(2πky/Ly)²]t)
                    cos(2πkx(x−x₀)/Lx+φx) cos(2πky(y−y₀)/Ly+φy). Zero wave
                    numbers and constant mode are retained.
                  </p>
                </details>
                <label>
                  I already have a complete reference u(x,y,t)
                  <input
                    aria-label="Complete reference formula"
                    value={cfg.complete}
                    placeholder="Optional expression to verify"
                    onChange={(e) => edit({ complete: e.target.value })}
                  />
                </label>
                {reference?.verification && (
                  <p>{JSON.stringify(reference.verification)}</p>
                )}
                <p>
                  Complete expressions are checked against the PDE, initial
                  field and periodic boundaries; they do not replace the
                  independent numerical solver.
                </p>
              </section>
              <section>
                <h3>Measurement details</h3>
                {sampleRecord &&
                typeof sampleRecord === "object" &&
                "placement" in sampleRecord ? (
                  <details>
                    <summary>Sampling provenance</summary>
                    <pre>{JSON.stringify(sampleRecord.placement, null, 2)}</pre>
                  </details>
                ) : null}
                <Numeric
                  label="Background coverage fraction"
                  value={backgroundFraction}
                  onChange={setBackgroundFraction}
                  min={0.1}
                  max={1}
                  step={0.05}
                />
                <label>
                  Chosen physical times
                  <input
                    aria-label="Chosen physical times"
                    value={selectedTimes}
                    onChange={(e) => setSelectedTimes(e.target.value)}
                  />
                </label>
                <label>
                  <input
                    type="checkbox"
                    checked={adaptive}
                    onChange={(e) => setAdaptive(e.target.checked)}
                  />
                  Time-adaptive positions (across chosen times only)
                </label>
                <p>
                  N is the total scalar count. Fixed sensors keep their
                  locations, designed at the current time; a partial final time
                  cycle is allowed. Purple hollow markers are candidates; white
                  dots are committed measurements. Heat focus uses reference
                  contrast, including cold regions, so positions convey
                  reference information.
                </p>
                <label>
                  Pattern
                  <select
                    aria-label="Measurement pattern"
                    value={pattern}
                    onChange={(e) => {
                      setPattern(e.target.value);
                      setPlacing(e.target.value === "manual");
                      if (e.target.value === "manual") setPlacement("manual");
                      sampleEpoch.current++;
                    }}
                  >
                    <option value="current">N at current physical time</option>
                    <option value="across">
                      N total across chosen / continuous times
                    </option>
                    <option value="fixed">
                      N total, fixed sensors over time
                    </option>
                    <option value="manual">
                      N manual points at current time
                    </option>
                  </select>
                </label>
                {
                  <Numeric
                    label={"Measurement seed"}
                    value={sampleSeed}
                    onChange={(v) => {
                      setSampleSeed(v);
                      invalidateRows();
                    }}
                    min={0}
                    max={2 ** 32 - 1}
                    step={1}
                  />
                }
                <label>
                  <input
                    type="checkbox"
                    checked={pointsOnly}
                    onChange={(e) => {
                      setPointsOnly(e.target.checked);
                      if (e.target.checked) {
                        setPlacement("random");
                        setPlacing(false);
                      }
                      invalidateRows();
                    }}
                  />
                  Measurements only (no reference comparison)
                </label>
                {pointsOnly && (
                  <div>
                    <Numeric
                      label="Observed value"
                      value={manualValue}
                      onChange={setManualValue}
                      min={-1e6}
                      max={1e6}
                    />
                    <Numeric
                      label="Observation x"
                      value={manualX}
                      onChange={setManualX}
                      min={cfg.origins[0]}
                      max={cfg.origins[0] + cfg.lengths[0]}
                    />
                    <Numeric
                      label="Observation y"
                      value={manualY}
                      onChange={setManualY}
                      min={cfg.origins[1]}
                      max={cfg.origins[1] + cfg.lengths[1]}
                    />
                    <button
                      onClick={() => {
                        const next = [
                          ...rows,
                          {
                            field: measurementQuantity,
                            x: manualX,
                            y: manualY,
                            time: physicalTime,
                            value: manualValue,
                            sigma: noise,
                          },
                        ];
                        invalidateResult();
                        setRows(next);
                        setSampleRecord({
                          rows: next,
                          source: "manual observed values",
                          reference_assisted: false,
                        });
                      }}
                    >
                      Add observed value
                    </button>
                  </div>
                )}
                <label>
                  Import observations (JSON array of field, x, y, time, value,
                  sigma)
                  <input
                    type="file"
                    aria-label="Import measurements"
                    accept=".json"
                    onChange={(e) => {
                      const f = e.target.files?.[0];
                      if (!f) return;
                      if (f.size > 2_000_000) {
                        setError("Observation file exceeds 2 MB.");
                        return;
                      }
                      void f
                        .text()
                        .then((s) => {
                          const imported = JSON.parse(s);
                          const r = Array.isArray(imported)
                            ? imported
                            : imported.rows;
                          if (
                            !Array.isArray(r) ||
                            r.length < 1 ||
                            r.length > 2000 ||
                            r.some(
                              (p) =>
                                p.field !== "u" ||
                                p.time < 0 ||
                                p.time > cfg.end ||
                                p.sigma < 0 ||
                                p.x < cfg.origins[0] ||
                                p.x >= cfg.origins[0] + cfg.lengths[0] ||
                                p.y < cfg.origins[1] ||
                                p.y >= cfg.origins[1] + cfg.lengths[1] ||
                                !["x", "y", "time", "value", "sigma"].every(
                                  (k) => Number.isFinite(p[k]),
                                ),
                            )
                          )
                            throw new Error(
                              "Import 1–2000 scalar u observations with finite x, y, time, value and sigma.",
                            );
                          invalidateRows();
                          setRows(r);
                          setPointsOnly(true);
                          setSampleRecord(
                            Array.isArray(imported)
                              ? { rows: r, source: "user import" }
                              : imported,
                          );
                        })
                        .catch((e) => setError(e.message));
                    }}
                  />
                </label>
                <button
                  disabled={!rows.length}
                  onClick={() =>
                    download("heat-measurements.json", sampleRecord || { rows })
                  }
                >
                  Export observations & metadata
                </button>
                {rows.length > 0 && (
                  <details>
                    <summary>{rows.length} saved scalar observations</summary>
                    <div className="measurement-table">
                      <table>
                        <thead>
                          <tr>
                            {["x", "y", "time", "value", "sigma"].map((k) => (
                              <th key={k}>{k}</th>
                            ))}
                          </tr>
                        </thead>
                        <tbody>
                          {rows.map((r, i) => (
                            <tr key={i}>
                              {[r.x, r.y, r.time, r.value, r.sigma].map(
                                (v, j) => (
                                  <td key={j}>{v.toPrecision(4)}</td>
                                ),
                              )}
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </details>
                )}
                <p>
                  Locations/noise use independent seeded streams. Clean values
                  remain evaluation-only on disk. Scrubbing never changes saved
                  observations. Dots show observations at the displayed physical
                  time.
                </p>
              </section>
              <section>
                <h3>Model settings</h3>
                <label>
                  Model output detail
                  <select
                    aria-label="Model output detail"
                    value={detail}
                    onChange={(e) => {
                      setDetail(+e.target.value);
                      invalidateResult();
                    }}
                  >
                    <option value={16}>16 × 16</option>
                    <option value={32}>32 × 32</option>
                    <option value={64}>64 × 64</option>
                  </select>
                </label>
                <p>
                  More pixels do not guarantee more learned physical detail.
                  Training grids and reference grids are separate.
                </p>
                <label>
                  Inference device
                  <select
                    aria-label="Inference device"
                    value={device}
                    onChange={(e) => {
                      setDevice(e.target.value);
                      invalidateResult();
                    }}
                  >
                    <option value="cpu">CPU</option>
                    <option value="mps">Apple MPS (if available)</option>
                    <option value="cuda">CUDA (if available)</option>
                    <option value="auto">Automatic</option>
                  </select>
                </label>
                {
                  <Numeric
                    label={"Heun steps"}
                    value={steps}
                    onChange={(v) => {
                      setSteps(v);
                      invalidateResult();
                    }}
                    min={1}
                    max={512}
                    step={1}
                  />
                }
                {
                  <Numeric
                    label={"Reconstruction samples"}
                    value={particles}
                    onChange={(v) => {
                      setParticles(v);
                      invalidateResult();
                    }}
                    min={1}
                    max={16}
                    step={1}
                  />
                }
                <p>
                  Deterministic latent conditional flow. Samples use seed 91.
                  Ensemble spread is not calibrated confidence.
                </p>
                <button onClick={() => importInput.current?.click()}>
                  Import another model
                </button>
                {result && (
                  <button
                    onClick={() => {
                      setResult(null);
                      setIndex(0);
                    }}
                  >
                    Return to single field
                  </button>
                )}
              </section>
            </div>
          </details>
        </main>
      )}
      <input
        ref={importInput}
        className="sr-only"
        type="file"
        aria-label="Import heat model"
        accept=".pt"
        onChange={(e) => {
          const f = e.target.files?.[0];
          if (f) void importModel(f);
          e.target.value = "";
        }}
      />
      <footer className="heat-footer">
        <span>PDEFlow · Interactive physics, honest reconstructions.</span>
        <span>2D heat / wave · nondimensional units</span>
      </footer>
    </div>
  );
}
