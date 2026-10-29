import { CoordinateReconstruction } from "./CoordinateModel";
import {
  useCallback,
  useEffect,
  useRef,
  useState,
  lazy,
  Suspense,
} from "react";
import {
  Activity,
  ArrowRight,
  CheckCircle2,
  Download,
  FlaskConical,
  Layers3,
  Plus,
  RefreshCw,
  Trash2,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { FKControls, FKPanel } from "./FKPanel";
import { defaultFK } from "./fkConfig";
import { ProcessControls, ProcessPanel } from "./ProcessPanel";
import { ReferenceWorkflow } from "./ReferenceWorkflow";
import { EvaluationPanel } from "./EvaluationPanel";
import { FieldDiagnostics } from "./FieldDiagnostics";
import { FieldPlot } from "./FieldPlot";
import { api, backendURL, saveFile, RequestError } from "./types";
import { RequestSummary } from "./RequestSummary";
import { InputPreview, SavedInputPreview } from "./InputPreview";
import { LiveDiagnostics } from "./LiveDiagnostics";
import type { Preview } from "./InputPreview";
import { ModelsTraining } from "./ModelsTraining";
import type { TrainingSelection } from "./TrainModel";
import { RunComparison } from "./RunComparison";
import { ResultCharts } from "./ScienceCharts";
import { readDraft, stable } from "./science";
import { insertPreset, importMathematics } from "./mathematics";
import type { ParseReport, Preset } from "./mathematics";
import { SymbolicEditor } from "./SymbolicEditor";
import { emptyMathematics } from "./mathematics";
import type { MathematicalRequest } from "./mathematics";
import type { Job, JobResult, Model, Problem, Row, Slice } from "./types";

const terminal = ["succeeded", "failed", "cancelled", "interrupted"];
const ScientificWorkspace = lazy(() =>
  import("./ScientificWorkspace").then((m) => ({
    default: m.ScientificWorkspace,
  })),
);
export default function App() {
  const [epoch, setEpoch] = useState(0);
  return <Workspace key={epoch} onClear={() => setEpoch((n) => n + 1)} />;
}
function Workspace({ onClear }: { onClear: () => void }) {
  const [saved] = useState(readDraft);
  const [clearOpen, setClearOpen] = useState(false),
    [errorDetails, setErrorDetails] = useState<unknown>(null),
    [captureGeneration, setCaptureGeneration] = useState(true);
  function clearWorkspace() {
    for (const key of Object.keys(localStorage))
      if (key.startsWith("pdeflow.")) localStorage.removeItem(key);
    onClear();
  }
  const [parseReport, setParseReport] = useState<ParseReport | null>(null),
    [reviewed, setReviewed] = useState<boolean>(saved?.reviewed ?? false),
    [selectedRow, setSelectedRow] = useState<number | null>(null),
    [rowIssues, setRowIssues] = useState<Preview["issues"]>([]),
    [guide, setGuide] = useState(false),
    [draftSaveError, setDraftSaveError] = useState(""),
    [resetToken] = useState(0);
  const [mathBlockReason, setMathBlockReason] = useState("");
  const [previousMath, setPreviousMath] = useState<MathematicalRequest | null>(
    null,
  );
  const [section, setSection] = useState(() => {
      try {
        const value = localStorage.getItem("pdeflow.view.v1");
        return value &&
          [
            "Problem",
            "Observations",
            "Reconstruction",
            "Results",
            "Runs",
            "Models",
          ].includes(value)
          ? value
          : "Problem";
      } catch {
        return "Problem";
      }
    }),
    [models, setModels] = useState<Model[]>([]),
    [modelId, setModelId] = useState(""),
    [modelOverride, setModelOverride] = useState(saved?.modelOverride ?? ""),
    [problem, setProblem] = useState<Problem>({
      schema_version: 1,
      family: "custom",
      grid: emptyMathematics().domain,
      fields: [{ name: "u", units: "nondimensional", role: "unknown" }],
      known_parameters: {},
      operator_spec: {},
      unknown_parameters: [],
      initial_condition: "unknown",
    }),
    [draftError, setDraftError] = useState("");
  const [mathRequest, setMathRequest] = useState<MathematicalRequest>(() => {
      try {
        return saved?.math
          ? importMathematics(JSON.stringify(saved.math))
          : emptyMathematics();
      } catch {
        return emptyMathematics();
      }
    }),
    [mathReady, setMathReady] = useState(false);
  const [comparisonPolicy, setComparisonPolicy] = useState("native"),
    [sharedComparisonTimes, setSharedComparisonTimes] = useState(false);
  const [observationInfo, setProblemInfo] = useState("");
  const [synthetic, setSynthetic] = useState(saved?.synthetic ?? false);
  const [rows, setRows] = useState<Row[]>(
      saved?.math?.measurements?.rows ?? [],
    ),
    [noise, setNoise] = useState(saved?.math?.measurements?.noise_std ?? 0.03),
    [referenceId, setReferenceId] = useState<string | null>(
      saved?.referenceId ?? null,
    ),
    [inference, setInference] = useState<Record<string, unknown>>(
      saved?.inference ?? {},
    ),
    [mode, setMode] = useState(saved?.mode ?? "observation_pde"),
    [device, setDevice] = useState(saved?.device ?? "cpu"),
    [backends, setBackends] = useState<Record<string, boolean>>({ cpu: true });
  const [error, setError] = useState(""),
    [backendOnline, setBackendOnline] = useState<boolean | null>(null),
    [busy, setBusy] = useState(false),
    [validation, setValidation] = useState<{
      valid: boolean;
      sample_storage_bytes: number;
      warnings: string[];
      resolved: unknown;
    } | null>(null),
    [job, setJob] = useState<Job | null>(null),
    [jobs, setJobs] = useState<Job[]>([]),
    [result, setResult] = useState<JobResult | null>(null),
    [slice, setSlice] = useState<Slice | null>(null),
    [spreadSlice, setSpreadSlice] = useState<Slice | null>(null),
    [showObservations, setShowObservations] = useState(true),
    [showReference, setShowReference] = useState(true),
    [timeIndex, setTimeIndex] = useState(0),
    [sliceAxis, setSliceAxis] = useState(2),
    [sliceIndex, setSliceIndex] = useState(0),
    [sampleIndex, setSampleIndex] = useState(-1),
    [fieldIndex, setFieldIndex] = useState(0);
  const submitting = useRef(false),
    token = useRef(crypto.randomUUID());
  const updateProblem = useCallback((p: Problem) => {
    setProblem(p);
    setDraftError("");
    setValidation(null);
    token.current = crypto.randomUUID();
  }, []);
  const onMathInvalid = useCallback((reason?: string) => {
    setMathBlockReason(reason ?? "");
    setParseReport(null);
    setMathReady(false);
    setValidation(null);
  }, []);
  const onMathParsed = useCallback(
    (p: Problem | null) => {
      setMathReady(!!p);
      if (p) updateProblem(p);
    },
    [updateProblem],
  );
  const completeMath = {
    ...mathRequest,
    measurements: { ...mathRequest.measurements, rows, noise_std: noise },
  };
  const persistedKey = JSON.stringify({
    version: 2,
    math: completeMath,
    inference,
    mode,
    device,
    modelOverride,
    reviewed,
    referenceId,
    synthetic,
  });
  // Storage is an external system; the error state reports its synchronous quota failure.
  useEffect(() => {
    try {
      localStorage.setItem("pdeflow.draft.v2", persistedKey);
      // eslint-disable-next-line react-hooks/set-state-in-effect
      setDraftSaveError("");
    } catch {
      setDraftSaveError(
        "This draft is too large for local storage. Export the complete request to preserve it.",
      );
    }
  }, [persistedKey]);
  const changeMath = (next: MathematicalRequest) => {
    setPreviousMath(completeMath);
    setReviewed(false);
    setMathRequest(next);
    setRows(next.measurements.rows);
    setNoise(next.measurements.noise_std);
    setReferenceId(null);
    onMathInvalid();
  };
  const refreshModels = useCallback(() => {
    return Promise.all([
      api<Model[]>("/models"),
      api<{ backends: Record<string, boolean> }>("/health"),
    ])
      .then(([m, h]) => {
        setBackendOnline(true);
        setModels(m);
        setBackends(h.backends);
        if (m.length) {
          setInference((current) =>
            Object.keys(current).length
              ? current
              : { ...m[0].config.inference, controller: {} },
          );
        }
      })
      .catch(() => setBackendOnline(false));
  }, []);
  useEffect(() => {
    void refreshModels();
  }, [refreshModels]);
  const loadJobs = useCallback(
    () =>
      api<Job[]>("/jobs")
        .then(setJobs)
        .catch((e) => {
          if (String(e).includes("Backend unavailable"))
            setBackendOnline(false);
          else setError(String(e));
        }),
    [],
  );
  useEffect(() => {
    void loadJobs();
  }, [loadJobs]);
  useEffect(() => {
    try {
      localStorage.setItem("pdeflow.view.v1", section);
    } catch {
      /* Draft export remains available. */
    }
  }, [section]);
  useEffect(() => {
    const id = localStorage.getItem("pdeflow.selected-job.v1");
    if (id && /^[a-f0-9]{32}$/.test(id))
      void api<Job>("/jobs/" + id)
        .then((j) => {
          if (j.id === id) {
            setJob(j);
          }
        })
        .catch(() => {});
  }, []);
  useEffect(() => {
    if (job?.id) localStorage.setItem("pdeflow.selected-job.v1", job.id);
  }, [job?.id]);
  useEffect(() => {
    if (!job || terminal.includes(job.status)) return;
    const id = setInterval(
      () =>
        api<Job>("/jobs/" + job.id)
          .then((j) => {
            setJob(j);
            if (terminal.includes(j.status)) {
              void loadJobs();
              if (j.status === "succeeded") setSection("Results");
            }
          })
          .catch((e) => setError(String(e))),
      350,
    );
    return () => clearInterval(id);
  }, [job, loadJobs]);
  useEffect(() => {
    if (!job || job.status !== "succeeded") return;
    let live = true;
    api<JobResult>("/jobs/" + job.id + "/result")
      .then((r) => {
        if (live) setResult(r);
      })
      .catch((e) => setError(String(e)));
    return () => {
      live = false;
    };
  }, [job]);
  useEffect(() => {
    if (!result || !job || job.status !== "succeeded") return;
    let live = true;
    api<Slice>(
      `/jobs/${job.id}/slice?time_index=${timeIndex}&axis=${sliceAxis}&index=${sliceIndex}&sample=${sampleIndex}&field_index=${fieldIndex}`,
    )
      .then((s) => {
        if (live) setSlice(s);
      })
      .catch((e) => setError(String(e)));
    return () => {
      live = false;
    };
  }, [result, job, timeIndex, sliceAxis, sliceIndex, sampleIndex, fieldIndex]);
  useEffect(() => {
    if (!result || !job || job.status !== "succeeded") return;
    let live = true;
    void api<Slice>(
      `/jobs/${job.id}/slice?time_index=${timeIndex}&axis=${sliceAxis}&index=${sliceIndex}&sample=-2&field_index=${fieldIndex}`,
    )
      .then((v) => {
        if (live) setSpreadSlice(v);
      })
      .catch(() => {
        if (live) setSpreadSlice(null);
      });
    return () => {
      live = false;
    };
  }, [result, job, timeIndex, sliceAxis, sliceIndex, fieldIndex]);
  const invalidate = () => {
    setValidation(null);
    token.current = crypto.randomUUID();
  };
  const changeInf = (key: string, value: unknown) => {
    setInference((v) => ({ ...v, [key]: value }));
    invalidate();
  };
  const request = () => ({
    model_selection: {
      mode: modelOverride ? "explicit" : "auto",
      id: modelOverride || null,
    },
    problem,
    rows,
    noise_std: noise,
    tolerance: Number(inference.noise_floor ?? 0.1),
    inference: {
      ...inference,
      snapshots: captureGeneration
        ? {
            max_frames: 9,
            max_side: 48,
            max_particles: 4,
            field_indices: problem.fields.map((_, i) => i),
            time_indices: problem.grid.times.map((_, i) => i).slice(0, 8),
          }
        : null,
    },
    runtime: { device, dtype: "float32", precision: "full", threads: 2 },
    mode,
    submission_token: token.current,
    mathematical_request: completeMath,
  });
  async function action(fn: () => Promise<void>) {
    setBusy(true);
    setError("");
    setErrorDetails(null);
    try {
      await fn();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      if (e instanceof RequestError) setErrorDetails(e.details);
    } finally {
      setBusy(false);
    }
  }
  async function validate() {
    if (draftError) throw new Error(draftError);
    if (!mathReady)
      throw new Error(
        mathBlockReason ||
          (parseReport
            ? "The current equation or supplied inputs are unsupported. Review the equation diagnostics before execution."
            : "Model selection and current problem validation are still loading. Wait for parsing to finish."),
      );
    const v = await api<{
      valid: boolean;
      sample_storage_bytes: number;
      warnings: string[];
      resolved: unknown;
    }>("/reconstruction/validate", request());
    setValidation(v);
    if (
      v.resolved &&
      typeof v.resolved === "object" &&
      "model_id" in v.resolved
    )
      setModelId(String(v.resolved.model_id));
  }
  const [coordinateSession, setCoordinateSession] = useState(0);
  const [selectedCoordinateModel, setSelectedCoordinateModel] = useState<string>();
  function applyTrainingSelection(selection: TrainingSelection, navigate = true) {
    if (selection.adopt_training_problem) changeMath(selection.suggested_request);
    if (selection.model === "resolution-flexible") {
      setModelOverride("");
      setSelectedCoordinateModel(selection.id);
      setCoordinateSession(v => v + 1);
    } else {
      setModelOverride(selection.id);
      setModelId(selection.id);
    }
    if (selection.selected_at) localStorage.setItem("pdeflow.training.selection-applied", selection.selected_at);
    invalidate();
    if (navigate) setSection("Reconstruction");
  }
  const startupSelection = useRef(applyTrainingSelection);
  useEffect(() => {
    void api<TrainingSelection | null>("/training/selection").then(selection => {
      if (selection?.selected_at && localStorage.getItem("pdeflow.training.selection-applied") !== selection.selected_at) {
        // Apply an explicit CLI model choice while preserving the current draft.
        // The ordinary Use-with-training-problem button owns layout adoption.
        // Merely finishing training never writes this selection record.
        startupSelection.current({...selection, adopt_training_problem:false}, false);
      }
    }).catch(() => { /* Normal offline banner remains authoritative. */ });
  }, []);
  function reopen(j: Job) {
    if (j.summary?.mode === "learned_observation_conditioning") {
      localStorage.setItem("pdeflow.coordinate.job", JSON.stringify(j));
      setCoordinateSession((v) => v + 1);
      setSection("Reconstruction");
      return;
    }
    setJob(j);
    setResult(null);
    setSlice(null);
    setTimeIndex(0);
    setSliceIndex(0);
    setSampleIndex(-1);
    setFieldIndex(0);
    setSection(terminal.includes(j.status) ? "Results" : "Reconstruction");
  }
  const model = models.find((m) => m.id === modelId),
    dimension = mathRequest.domain.shape.length,
    axes = ["x", "y", "z"].slice(0, dimension);
  const rowKeys = [
    "field",
    ...axes,
    ...(mathRequest.domain.times.length ? ["time"] : []),
    "value",
  ];
  async function importObservations(text: string, format: string) {
    if (!problem) throw new Error("Choose a problem");
    const response = await api<{ rows: Row[] }>("/observations/validate", {
      problem,
      text,
      format,
      noise_std: noise,
      tolerance: Number(inference.noise_floor ?? 0.1),
    });
    setRows(response.rows);
    setMathRequest((v) => ({
      ...v,
      measurements: { ...v.measurements, status: "known" },
    }));
    setSynthetic(false);
    invalidate();
  }
  function exportCSV() {
    saveFile(
      "observations.csv",
      [
        rowKeys.join(","),
        ...rows.map((r) => rowKeys.map((k) => r[k]).join(",")),
      ].join("\n"),
    );
  }
  const fkSupported = Boolean(
    model?.steering?.fk &&
    problem &&
    model.problem.family === problem.family &&
    JSON.stringify(model.problem.grid) === JSON.stringify(problem.grid),
  );
  const running = job && !terminal.includes(job.status);
  return (
    <TooltipProvider>
      <div className="workbench">
        <aside className="sidebar">
          <a
            className="brand"
            href="#"
            onClick={(e) => {
              e.preventDefault();
              setSection("Problem");
            }}
          >
            <span className="brand-icon">
              <Layers3 size={23} />
            </span>
            <span>
              PDEFlow<small>SCIENTIFIC WORKBENCH</small>
            </span>
          </a>
          <div className="workspace-name">
            <span className="status-dot" /> LOCAL RESEARCH
          </div>
          <nav aria-label="Main navigation">
            {[
              ["Workspace", "Problem"],
              ["Runs", "Runs"],
              ["Models & Training", "Models"],
            ].map(([label, target]) => (
              <Button
                key={target}
                variant="ghost"
                aria-current={
                  target === "Problem"
                    ? !["Runs", "Models"].includes(section)
                      ? "page"
                      : undefined
                    : section === target
                      ? "page"
                      : undefined
                }
                onClick={() => setSection(target)}
              >
                {label}
              </Button>
            ))}
          </nav>
          <div className="sidebar-bottom">
            <FlaskConical size={20} />
            <p>
              Specialist models.
              <br />
              Explicit physical constraints.
            </p>
            <small>
              Periodic heat · 1D–3D
              <br />
              Heat + Darcy + chemistry execute
            </small>
          </div>
        </aside>
        <main>
          <header>
            <div className="breadcrumb">
              Workspace <span>/</span>{" "}
              {section === "Models" ? "Models & Training" : section}
            </div>
            <Badge variant="outline">
              {backendOnline === true && <span className="status-dot" />}
              {backendOnline === false
                ? "Backend unavailable"
                : backendOnline === null
                  ? "Checking backend"
                  : "Local Python API"}
            </Badge>
          </header>
          {!["Runs", "Models"].includes(section) && (
            <>
              <div className="workspace-stages" aria-label="Workspace stages">
                {[
                  ["Define problem", "Problem"],
                  ["Add information", "Observations"],
                  [
                    "Reconstruct & inspect",
                    result ? "Results" : "Reconstruction",
                  ],
                ].map(([label, target], i) => (
                  <Button
                    key={i}
                    variant={
                      section === target ||
                      (i === 2 &&
                        ["Results", "Reconstruction"].includes(section))
                        ? "default"
                        : "outline"
                    }
                    aria-current={section === target ? "step" : undefined}
                    onClick={() => setSection(target)}
                  >
                    <span>{i + 1}</span>
                    {label}
                  </Button>
                ))}
              </div>
              <RequestSummary
                value={completeMath}
                report={parseReport}
                reviewed={reviewed}
                onEdit={(stage, target) => {
                  setSection(stage);
                  setTimeout(() => {
                    const element = document.getElementById(target);
                    let parent = element;
                    while (parent) {
                      if (parent instanceof HTMLDetailsElement)
                        parent.open = true;
                      parent = parent.parentElement;
                    }
                    element?.scrollIntoView({
                      behavior: "smooth",
                      block: "start",
                    });
                  }, 0);
                }}
              />
              <div className="draft-actions">
                <small>
                  {draftSaveError ||
                    "Draft saved on this device. Preset defaults and checkpoint suggestions still need review."}
                </small>
                <Button
                  size="sm"
                  variant="ghost"
                  onClick={() => {
                    setGuide((v) => !v);
                  }}
                >
                  Guided example
                </Button>
                <Button
                  size="sm"
                  variant="outline"
                  onClick={() => setClearOpen(true)}
                >
                  Clear workspace
                </Button>
                <Dialog open={clearOpen} onOpenChange={setClearOpen}>
                  <DialogContent>
                    <DialogHeader>
                      <DialogTitle>Clear the current workspace?</DialogTitle>
                      <DialogDescription>
                        Discard the current equation, inputs, observations,
                        reference/model/result selections and persisted draft.
                        Saved runs, checkpoints, reference artifacts and
                        training history remain available.
                      </DialogDescription>
                    </DialogHeader>
                    {running ? (
                      <>
                        <Button onClick={clearWorkspace}>
                          Keep job running and clear editor
                        </Button>
                        <Button
                          variant="outline"
                          onClick={() =>
                            void action(async () => {
                              await api(`/jobs/${job!.id}/cancel`, {});
                              clearWorkspace();
                            })
                          }
                        >
                          Cancel job, then clear
                        </Button>
                      </>
                    ) : (
                      <Button onClick={clearWorkspace}>
                        Clear current draft
                      </Button>
                    )}
                    <Button variant="ghost" onClick={() => setClearOpen(false)}>
                      Return without clearing
                    </Button>
                  </DialogContent>
                </Dialog>
                <Button
                  size="sm"
                  variant="ghost"
                  onClick={() => {
                    setClearOpen(true);
                  }}
                >
                  Reset draft
                </Button>
                {previousMath && (
                  <Button
                    size="sm"
                    variant="ghost"
                    onClick={() => changeMath(previousMath!)}
                  >
                    Undo last problem edit
                  </Button>
                )}
              </div>
              {guide && (
                <Card className="guided-example">
                  <CardHeader>
                    <CardTitle>A real heat reconstruction</CardTitle>
                    <CardDescription>
                      No fake fields or precomputed success is inserted.
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <ol>
                      <li>
                        Insert the editable heat equation and inspect its
                        symbols/domain.
                      </li>
                      <li>
                        Add information: generate real synthetic measurements or
                        import your own.
                      </li>
                      <li>
                        Review the compatible model, then validate and
                        reconstruct.
                      </li>
                      <li>
                        Inspect the returned fields; attach truth only for
                        evaluation.
                      </li>
                    </ol>
                    <Button
                      disabled={mathRequest.equations.some((e) => e.trim())}
                      onClick={() =>
                        void action(async () => {
                          const catalog = await api<Preset[]>(
                            "/mathematics/presets",
                          );
                          const heat = catalog.find((p) => p.id === "heat");
                          if (heat) {
                            changeMath(insertPreset(emptyMathematics(), heat));
                            setSection("Problem");
                          }
                        })
                      }
                    >
                      Load supported heat example
                    </Button>
                    <Button variant="ghost" onClick={() => setGuide(false)}>
                      Dismiss example
                    </Button>
                    <p>
                      A nonempty draft is preserved. Use equation preset
                      insertion to review replacements.
                    </p>
                  </CardContent>
                </Card>
              )}
            </>
          )}
          <div className="page-heading">
            <div>
              <p className="eyebrow">RECONSTRUCTION LAB</p>
              <h1>
                {section === "Problem"
                  ? "Define the physical problem"
                  : section === "Observations"
                    ? "Bring your measurements"
                    : section === "Reconstruction"
                      ? "Configure & reconstruct"
                      : section === "Results"
                        ? "Inspect the reconstruction"
                        : section === "Models" ? "Train and use PDE models" : "Research run history"}
              </h1>
              <p>
                {section === "Problem"
                  ? "Specify the equation, domain and unknowns. Each choice is checked against the model."
                  : section === "Results"
                    ? "Physical fields and diagnostics from actual solver outputs."
                    : section === "Models" ? "Choose a model and budget, inspect validation, then use the resulting checkpoint." : "A reproducible path from sparse observations to candidate fields."}
              </p>
            </div>
            <Badge variant="secondary">
              {section === "Results" && result
                ? result.request.problem.grid.shape.length
                : dimension}{" "}
              spatial dimensions
            </Badge>
          </div>
          <Button
            variant="outline"
            size="sm"
            onClick={() => void refreshModels()}
          >
            <RefreshCw size={14} /> Refresh checkpoints
          </Button>
          {backendOnline === false && (
            <Alert
              variant="destructive"
              role="alert"
              aria-label="Backend connection"
            >
              <AlertTitle>Backend unavailable.</AlertTitle>
              <AlertDescription>
                Expected API: {backendURL}. Start it with{" "}
                <code>pdeflow serve</code>, then refresh checkpoints.
              </AlertDescription>
            </Alert>
          )}
          {error && (
            <Alert variant="destructive" role="alert">
              <AlertTitle>Action needs attention</AlertTitle>
              <AlertDescription>
                {error}
                {errorDetails ? (
                  <details>
                    <summary>Technical request details</summary>
                    <pre>{JSON.stringify(errorDetails, null, 2)}</pre>
                  </details>
                ) : null}
                <Button variant="ghost" size="sm" onClick={() => setError("")}>
                  Dismiss
                </Button>
              </AlertDescription>
            </Alert>
          )}
          {section === "Models" && (
            <ModelsTraining
              problem={problem}
              models={models}
              onRefresh={() => void refreshModels()}
              onUse={applyTrainingSelection}
            />
          )}
          {section === "Results" && result && (
            <div className="result-context">
              <strong>
                Run {job?.id.slice(0, 8)} · immutable saved result
              </strong>
              <p>
                {stable({ math: completeMath, model: modelId }) !==
                stable({
                  math: result.request.mathematical_request,
                  model: result.request.model_id,
                })
                  ? "Inputs changed since this run. These results belong to the earlier request; your current draft remains separate."
                  : "Results correspond to the current problem and measurements."}
              </p>
              <Button
                variant="outline"
                onClick={() => setSection("Reconstruction")}
              >
                Adjust reconstruction settings
              </Button>
            </div>
          )}
          {!problem ? (
            <Card>
              <CardHeader>
                <CardTitle>No registered checkpoints</CardTitle>
                <CardDescription>
                  Configure model roots and create a checkpoint with the CLI.
                  Opening this workbench does not start training.
                </CardDescription>
              </CardHeader>
            </Card>
          ) : (
            <>
              {
                <div
                  className="definition-workspace"
                  style={{
                    display: section === "Problem" ? undefined : "none",
                  }}
                >
                  <SymbolicEditor
                    key={resetToken}
                    value={completeMath}
                    onChange={changeMath}
                    models={models}
                    modelId={modelOverride}
                    onModel={(id) => {
                      setModelOverride(id);
                      invalidate();
                    }}
                    onRoute={setModelId}
                    onParsed={onMathParsed}
                    onInvalid={onMathInvalid}
                    validated={!!validation && mathReady}
                    onReport={setParseReport}
                  />
                  {section === "Problem" && (
                    <div className="definition-preview">
                      <InputPreview
                        problem={mathReady ? problem : null}
                        rows={rows}
                        noise={noise}
                        selected={selectedRow}
                        onSelect={setSelectedRow}
                      />
                      <Button
                        variant="outline"
                        onClick={() => setReviewed(true)}
                      >
                        Confirm reviewed parameters, domain & conditions
                      </Button>
                      <p>
                        Review marks your intent; server compatibility checks
                        remain authoritative.
                      </p>
                    </div>
                  )}
                </div>
              }
              {section === "Observations" && (
                <div className="stack" id="observation-editor">
                  <InputPreview
                    problem={mathReady ? problem : null}
                    rows={rows}
                    noise={noise}
                    selected={selectedRow}
                    onSelect={setSelectedRow}
                    onIssues={setRowIssues}
                  />
                  <Card>
                    <CardHeader>
                      <CardTitle>
                        Measurements in physical coordinates
                      </CardTitle>
                      <CardDescription>
                        Required: {rowKeys.join(", ")}. On-grid coordinates and
                        saved times only. A measured zero is still observed.
                      </CardDescription>
                    </CardHeader>
                    <CardContent className="stack">
                      <div className="toolbar">
                        <Field label="Independent noise σ">
                          <Input
                            aria-label="Observation noise"
                            type="number"
                            min="0"
                            step=".01"
                            value={noise}
                            onChange={(e) => {
                              setNoise(Number(e.target.value));
                              invalidate();
                            }}
                          />
                        </Field>
                        <label className="file-button">
                          Import CSV / JSON
                          <input
                            aria-label="Import observations"
                            type="file"
                            accept=".csv,.json"
                            onChange={(e) => {
                              const file = e.target.files?.[0];
                              invalidate();
                              if (file)
                                void action(async () => {
                                  if (file.size > 1000000)
                                    throw new Error(
                                      "Observation file exceeds 1 MB",
                                    );
                                  await importObservations(
                                    await file.text(),
                                    file.name.endsWith(".json")
                                      ? "json"
                                      : "csv",
                                  );
                                });
                              e.target.value = "";
                            }}
                          />
                        </label>
                        <Button variant="outline" onClick={exportCSV}>
                          Export CSV
                        </Button>
                        <Button
                          variant="outline"
                          onClick={() => saveFile("observations.json", rows)}
                        >
                          Export JSON
                        </Button>
                        <Badge variant="secondary">
                          {rows.length} measurements
                        </Badge>
                        {synthetic && (
                          <Badge variant="outline">
                            Synthetic observations
                          </Badge>
                        )}
                      </div>
                      <div className="observation-table">
                        <table>
                          <thead>
                            <tr>
                              <th>Row / selection</th>
                              {rowKeys.map((k) => (
                                <th key={k}>{k}</th>
                              ))}
                              <th>Remove</th>
                            </tr>
                          </thead>
                          <tbody>
                            {rows.map((row, i) => (
                              <tr
                                key={i}
                                id={`measurement-row-${i}`}
                                data-selected={selectedRow === i}
                                aria-invalid={rowIssues.some(
                                  (e) => e.row === i,
                                )}
                                onFocus={() => setSelectedRow(i)}
                                onClick={() => setSelectedRow(i)}
                              >
                                <td>
                                  <Button
                                    size="sm"
                                    variant="ghost"
                                    aria-label={`Select observation row ${i + 1}`}
                                    onClick={() => setSelectedRow(i)}
                                  >
                                    {i + 1}
                                  </Button>
                                  {rowIssues
                                    .filter((e) => e.row === i)
                                    .map((e, j) => (
                                      <small role="alert" key={j}>
                                        {e.message}
                                      </small>
                                    ))}
                                </td>
                                {rowKeys.map((key) => (
                                  <td key={key}>
                                    {key === "field" ? (
                                      <select
                                        aria-label={`Row ${i + 1} field`}
                                        value={String(row[key])}
                                        onChange={(e) => {
                                          setRows(
                                            rows.map((r, j) =>
                                              j === i
                                                ? {
                                                    ...r,
                                                    [key]: e.target.value,
                                                  }
                                                : r,
                                            ),
                                          );
                                          invalidate();
                                        }}
                                      >
                                        {mathRequest.declarations
                                          .filter(
                                            (d) =>
                                              (d.kind === "field" ||
                                                d.kind === "coefficient") &&
                                              d.status === "unknown",
                                          )
                                          .map((f) => (
                                            <option key={f.name}>
                                              {f.name}
                                            </option>
                                          ))}
                                      </select>
                                    ) : (
                                      <Input
                                        aria-label={`Row ${i + 1} ${key}`}
                                        value={String(row[key] ?? "")}
                                        onChange={(e) => {
                                          setRows(
                                            rows.map((r, j) =>
                                              j === i
                                                ? {
                                                    ...r,
                                                    [key]: e.target.value,
                                                  }
                                                : r,
                                            ),
                                          );
                                          invalidate();
                                        }}
                                      />
                                    )}
                                  </td>
                                ))}
                                <td>
                                  <Button
                                    size="icon"
                                    variant="ghost"
                                    aria-label={`Remove row ${i + 1}`}
                                    onClick={() => {
                                      setRows(rows.filter((_, j) => i !== j));
                                      invalidate();
                                    }}
                                  >
                                    <Trash2 size={14} />
                                  </Button>
                                </td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                        {rows.length === 0 && (
                          <p className="empty">
                            Add a measurement, import a file, or explicitly
                            generate a synthetic demo.
                          </p>
                        )}
                      </div>
                      <div className="toolbar">
                        <Button
                          variant="outline"
                          onClick={() => {
                            setRows([
                              ...rows,
                              {
                                field:
                                  mathRequest.declarations.find(
                                    (d) =>
                                      (d.kind === "field" ||
                                        d.kind === "coefficient") &&
                                      d.status === "unknown",
                                  )?.name ?? "u",
                                ...Object.fromEntries(
                                  axes.map((a, j) => [
                                    a,
                                    problem.grid.domain.origins[j] +
                                      (problem.grid.convention ===
                                      "periodic_cell_centers"
                                        ? (0.5 *
                                            problem.grid.domain.lengths[j]) /
                                          problem.grid.shape[j]
                                        : 0),
                                  ]),
                                ),
                                ...(mathRequest.domain.times.length
                                  ? { time: mathRequest.domain.times[0] }
                                  : {}),
                                value: 0,
                              },
                            ]);
                            invalidate();
                          }}
                        >
                          <Plus size={15} />
                          Add measurement
                        </Button>
                        <Button
                          disabled={busy || !mathReady}
                          onClick={() =>
                            void action(async () => {
                              const v = await api<{ count: number }>(
                                "/observations/validate",
                                {
                                  problem,
                                  rows,
                                  noise_std: noise,
                                  tolerance: Number(
                                    inference.noise_floor ?? 0.1,
                                  ),
                                },
                              );
                              setProblemInfo(
                                `${v.count} valid on-grid measurements`,
                              );
                            })
                          }
                        >
                          Validate observations
                        </Button>
                      </div>
                    </CardContent>
                  </Card>
                  <p
                    className="information-note"
                    aria-label="Measurement coverage"
                  >
                    Measured fields:{" "}
                    {[...new Set(rows.map((r) => String(r.field)))].join(
                      ", ",
                    ) || "None"}
                    .
                    {problem.grid.temporal_kind !== "static" &&
                      ` Saved physical times represented: ${[...new Set(rows.map((r) => r.time))].sort((a, b) => Number(a) - Number(b)).join(", ") || "None"}.`}
                    {["x", "y", "z"].slice(0, dimension).map((axis) => {
                      const values = rows
                        .map((r) => Number(r[axis]))
                        .filter(Number.isFinite);
                      return values.length
                        ? ` ${axis} coverage: ${Math.min(...values).toPrecision(3)}–${Math.max(...values).toPrecision(3)}.`
                        : "";
                    })}
                  </p>
                  <p className="information-note">
                    Spatial coordinate y is distinct from the measurement vector
                    y. Noise σ is an independent Gaussian standard deviation;
                    guidance weights are separate from physical parameters.
                  </p>
                  {observationInfo && (
                    <p className="validation-note">{observationInfo}</p>
                  )}
                  <Card className="synthetic-card">
                    <CardHeader>
                      <CardTitle>
                        <FlaskConical size={19} /> Explicit synthetic reference
                        action
                      </CardTitle>
                      <CardDescription>
                        Creates a reference with the recognized specialist and
                        noisy sensors. Hidden fields are stored separately and
                        never sent to inference.
                      </CardDescription>
                    </CardHeader>
                    <CardContent className="toolbar">
                      <p>
                        Generates{" "}
                        {Math.min(
                          10,
                          problem.grid.shape.reduce((a, b) => a * b, 1),
                        )}{" "}
                        spatial sensor locations per field
                        {problem.grid.temporal_kind === "static"
                          ? ""
                          : " at each saved physical time"}
                        , using the noise σ specified above. Unmeasured regions
                        remain unknown.
                      </p>
                      <Button
                        disabled={busy || !mathReady}
                        onClick={() =>
                          void action(async () => {
                            const demo = await api<{
                              problem: Problem;
                              rows: Row[];
                              reference_id: string;
                            }>("/demos", {
                              model_id: modelId || null,
                              problem,
                              seed: 53,
                              sensors: Math.min(
                                10,
                                problem.grid.shape.reduce((a, b) => a * b, 1),
                              ),
                              noise_std: noise,
                            });
                            updateProblem(demo.problem);
                            setRows(demo.rows);
                            setMathRequest((v) => ({
                              ...v,
                              measurements: {
                                ...v.measurements,
                                status: "known",
                              },
                            }));
                            setSynthetic(true);
                            setReferenceId(demo.reference_id);
                            invalidate();
                            setProblemInfo(
                              "Synthetic observations generated; evaluation reference is detached from inference",
                            );
                          })
                        }
                      >
                        Generate synthetic observations
                      </Button>
                      {referenceId && (
                        <>
                          <Badge variant="outline">
                            Synthetic reference available for evaluation only
                          </Badge>
                          <Button
                            variant="ghost"
                            onClick={() => setReferenceId(null)}
                          >
                            Detach reference
                          </Button>
                        </>
                      )}
                    </CardContent>
                  </Card>
                </div>
              )}
              <ReferenceWorkflow
                math={completeMath}
                problem={problem}
                ready={mathReady}
                visible={["Problem", "Observations"].includes(section)}
                onSample={(v, id) => {
                  changeMath(v.mathematical_request);
                  setReferenceId(id);
                  setRows(v.rows);
                  setSynthetic(Boolean(id));
                  setProblemInfo(
                    id
                      ? "Noisy measurements adopted; full reference inputs are retained for evaluation only."
                      : "Measurements adopted without a reference; ground-truth errors are unavailable.",
                  );
                }}
                onReconstruct={() => setSection("Reconstruction")}
              />
              {section === "Reconstruction" && (
                <CoordinateReconstruction
                  key={coordinateSession}
                  selectedModel={selectedCoordinateModel}
                  ready={mathReady}
                  mathematicalRequest={completeMath}
                  problem={problem}
                  rows={rows}
                  referenceId={referenceId}
                  noiseStd={completeMath.measurements.noise_std}
                />
              )}
              {section === "Reconstruction" && (
                <div className="two-columns">
                  <Card>
                    <CardHeader>
                      <CardTitle>Generation & steering</CardTitle>
                      <CardDescription>
                        Generation steps integrate the learned flow. They do not
                        refine the physical grid.
                      </CardDescription>
                    </CardHeader>
                    <CardContent className="stack">
                      <div className="form-pair">
                        <Field
                          label={
                            ["fk", "importance"].includes(mode)
                              ? "Particles"
                              : "Independent samples"
                          }
                        >
                          <Input
                            aria-label="Sample count"
                            type="number"
                            min="1"
                            max={["fk", "importance"].includes(mode) ? 256 : 16}
                            value={Number(inference.particles ?? 2)}
                            onChange={(e) =>
                              changeInf("particles", Number(e.target.value))
                            }
                          />
                        </Field>

                        <Field label="Compute device">
                          <select
                            aria-label="Compute device"
                            value={device}
                            onChange={(e) => {
                              setDevice(e.target.value);
                              invalidate();
                            }}
                          >
                            {Object.entries(backends).map(([d, available]) => (
                              <option key={d} value={d} disabled={!available}>
                                {d.toUpperCase()}
                                {available ? "" : " · unavailable"}
                              </option>
                            ))}
                          </select>
                        </Field>
                      </div>
                      <Field label="Guidance mode">
                        <select
                          aria-label="Guidance mode"
                          value={mode}
                          onChange={(e) => {
                            setMode(e.target.value);
                            if (["fk", "importance"].includes(e.target.value))
                              setInference((v) => ({
                                ...v,
                                fk: v.fk ?? defaultFK,
                                controller: {
                                  ...((v.controller as object) ?? {}),
                                  max_particles: 256,
                                },
                              }));
                            invalidate();
                          }}
                        >
                          <option value="unguided">Unguided flow</option>
                          <option value="observation">
                            Observation guidance
                          </option>
                          <option value="observation_pde">
                            Observation + PDE guidance
                          </option>
                          <option value="fk" disabled={!fkSupported}>
                            Feynman–Kac steering
                          </option>
                          <option value="importance" disabled={!fkSupported}>
                            Terminal importance weighting
                          </option>
                        </select>
                      </Field>
                      <p className="method-help">
                        {
                          (
                            {
                              unguided:
                                "Independent candidates from the learned flow; measurements do not guide them.",
                              observation:
                                "Gradient corrections encourage agreement with the supplied measurements.",
                              observation_pde:
                                "Gradient corrections encourage agreement with measurements and the discretized equation.",
                              importance:
                                "Stochastic candidates receive terminal energy weights; output summaries respect those weights.",
                              fk: "Particles are candidate complete solutions, not physical molecules. Energy weighting and resampling change their population.",
                            } as Record<string, string>
                          )[mode]
                        }{" "}
                        More samples increase cost; empirical spread is not
                        calibrated uncertainty.
                      </p>
                      <details>
                        <summary>Numerical sampling settings</summary>
                        <div className="form-pair">
                          <Field label="Seed">
                            <Input
                              aria-label="Sampling seed"
                              type="number"
                              value={Number(inference.seed ?? 37)}
                              onChange={(e) =>
                                changeInf("seed", Number(e.target.value))
                              }
                            />
                          </Field>
                          <Field label="ODE stepper">
                            <select
                              aria-label="ODE stepper"
                              value={String(inference.stepper ?? "heun")}
                              onChange={(e) =>
                                changeInf("stepper", e.target.value)
                              }
                            >
                              <option value="euler">Euler</option>
                              <option value="heun">Heun</option>
                            </select>
                          </Field>
                          <Field label="Generation schedule">
                            <select
                              aria-label="Generation schedule"
                              value={String(inference.schedule ?? "uniform")}
                              onChange={(e) =>
                                changeInf("schedule", e.target.value)
                              }
                            >
                              <option value="uniform">Uniform</option>
                              <option value="quadratic">Quadratic</option>
                            </select>
                          </Field>
                          <Field label="Generation steps">
                            <Input
                              aria-label="Generation steps"
                              type="number"
                              min="1"
                              max="512"
                              value={Number(inference.steps ?? 12)}
                              onChange={(e) =>
                                changeInf("steps", Number(e.target.value))
                              }
                            />
                          </Field>
                        </div>
                        <p>
                          Generation coordinate s is not physical time τ. More
                          generation steps do not refine the spatial grid.
                        </p>
                      </details>
                      {["fk", "importance"].includes(mode) && (
                        <details>
                          <summary>FK particle & resampling settings</summary>
                          <FKControls
                            value={inference.fk}
                            onChange={(value) => changeInf("fk", value)}
                          />
                        </details>
                      )}
                      <ProcessControls
                        nativePoints={problem.grid.shape.reduce(
                          (a, b) => a * b,
                          1,
                        )}
                        value={inference.controller}
                        onChange={(value) => changeInf("controller", value)}
                      />
                      <details>
                        <summary>Advanced guidance controls</summary>
                        <div className="form-pair">
                          {[
                            "observation_strength",
                            "physics_strength",
                            "correction_strength",
                            "gradient_max_norm",
                            "noise_floor",
                          ].map((k) => (
                            <Field key={k} label={k.replaceAll("_", " ")}>
                              <Input
                                aria-label={k}
                                type="number"
                                step=".001"
                                value={Number(inference[k] ?? 0)}
                                onChange={(e) =>
                                  changeInf(k, Number(e.target.value))
                                }
                              />
                            </Field>
                          ))}
                        </div>
                      </details>
                      <Alert>
                        <AlertTitle>
                          {["fk", "importance"].includes(mode)
                            ? "Finite particle approximation"
                            : "Heuristic reconstruction"}
                        </AlertTitle>
                        <AlertDescription>
                          {["fk", "importance"].includes(mode)
                            ? "Finite particles approximate the stated numerical-base energy tilt. No gradient guidance drift is added; weighted spread is not calibrated posterior uncertainty."
                            : "Gradient corrections change the generated distribution. Independent sample spread is not calibrated posterior uncertainty."}
                        </AlertDescription>
                      </Alert>
                    </CardContent>
                  </Card>
                  <div className="stack">
                    <Card>
                      <CardHeader>
                        <CardTitle>Validate & launch</CardTitle>
                        <CardDescription>
                          {model?.name} · float32 · fixed saved-time window
                        </CardDescription>
                      </CardHeader>
                      <CardContent className="stack">
                        <p>
                          {rows.length} measurements · grid{" "}
                          {problem.grid.shape.join(" × ")} ·{" "}
                          {problem.grid.times.length} physical times
                        </p>
                        <Button
                          variant="outline"
                          disabled={
                            busy ||
                            !!running ||
                            (!mathReady && !parseReport && !mathBlockReason)
                          }
                          onClick={() => void action(validate)}
                        >
                          Validate complete request
                        </Button>
                        {!mathReady && !parseReport && !error && (
                          <p role="status">
                            {mathBlockReason ||
                              "Model selection and current problem validation are still loading."}
                          </p>
                        )}
                        {validation && (
                          <div className="validated">
                            <CheckCircle2 size={18} />
                            <p>
                              Compatible request
                              <br />
                              <small>
                                {(
                                  validation.sample_storage_bytes / 1024
                                ).toFixed(1)}{" "}
                                KiB sample arrays; full device memory is larger.
                              </small>
                              {validation.warnings.map((w) => (
                                <small key={w}>{w}</small>
                              ))}
                            </p>
                          </div>
                        )}
                        <label>
                          <input
                            type="checkbox"
                            checked={captureGeneration}
                            onChange={(e) => {
                              setCaptureGeneration(e.target.checked);
                              invalidate();
                            }}
                          />{" "}
                          Record bounded generation snapshots (up to9 frames,4
                          particles; display striding)
                        </label>
                        <Button
                          disabled={
                            busy || !!running || !validation || !mathReady
                          }
                          onClick={() => {
                            if (submitting.current) return;
                            submitting.current = true;
                            void action(async () => {
                              try {
                                await validate();
                                const j = await api<Job>("/jobs", request());
                                if (referenceId)
                                  await api(
                                    `/jobs/${j.id}/evaluation-context`,
                                    { reference_id: referenceId },
                                  );
                                setJob(j);
                                setResult(null);
                                setSlice(null);
                                void loadJobs();
                              } finally {
                                submitting.current = false;
                              }
                            });
                          }}
                        >
                          <Activity size={16} />
                          {busy ? "Working…" : "Run reconstruction"}
                        </Button>
                        <Button
                          variant="ghost"
                          onClick={() =>
                            saveFile(
                              "reconstruction-config.json",
                              validation?.resolved ?? request(),
                            )
                          }
                        >
                          <Download size={15} />
                          Export resolved request
                        </Button>
                      </CardContent>
                    </Card>
                    {job && (
                      <Card>
                        <CardHeader>
                          <CardTitle>Job {job.id.slice(0, 8)}</CardTitle>
                          <CardDescription>
                            {job.status} · {job.stage} ·{" "}
                            {job.stop_reason ?? "verification pending"}
                          </CardDescription>
                        </CardHeader>
                        <CardContent>
                          <Progress value={job.progress * 100} />
                          <LiveDiagnostics job={job} />
                          <p className="small-text">
                            {job.completed_steps ?? 0} /{" "}
                            {job.total_steps ?? (inference.steps as number)}{" "}
                            completed generation steps
                          </p>
                          {job.process_event && (
                            <details>
                              <summary>Latest physics/controller check</summary>
                              <pre>
                                {JSON.stringify(job.process_event, null, 2)}
                              </pre>
                            </details>
                          )}
                          {running && (
                            <Button
                              variant="destructive"
                              onClick={() =>
                                void action(async () => {
                                  setJob(
                                    await api<Job>(
                                      "/jobs/" + job.id + "/cancel",
                                      {},
                                    ),
                                  );
                                  void loadJobs();
                                })
                              }
                            >
                              Cancel job
                            </Button>
                          )}
                          {job.error && <p role="alert">{job.error.message}</p>}
                        </CardContent>
                      </Card>
                    )}
                  </div>
                </div>
              )}
              {section === "Results" &&
                (!result ? (
                  <Card>
                    <CardHeader>
                      <CardTitle>
                        {job
                          ? `Job ${job.status}`
                          : "No reconstruction selected"}
                      </CardTitle>
                      <CardDescription>
                        {job?.error?.message ??
                          "Run a compatible request or reopen a completed run from history."}
                      </CardDescription>
                    </CardHeader>
                  </Card>
                ) : (
                  <div className="stack">
                    <Suspense
                      fallback={<p>Loading linked mathematical workspace…</p>}
                    >
                      <ScientificWorkspace
                        key={job!.id}
                        jobId={job!.id}
                        problem={result.request.problem}
                        field={fieldIndex}
                        sample={sampleIndex}
                        time={timeIndex}
                        onTime={setTimeIndex}
                      />
                    </Suspense>
                    <Card>
                      <CardContent className="results-controls">
                        <Field
                          label={
                            result.request.problem.family === "chemistry"
                              ? "Species"
                              : "Physical field"
                          }
                        >
                          <select
                            aria-label="Physical field"
                            value={fieldIndex}
                            onChange={(e) =>
                              setFieldIndex(Number(e.target.value))
                            }
                          >
                            {result.request.problem.fields.map((f, j) => (
                              <option key={f.name} value={j}>
                                {f.name}
                              </option>
                            ))}
                          </select>
                        </Field>
                        {result.request.problem.grid.temporal_kind ===
                        "static" ? (
                          <p>Static fields · no physical time</p>
                        ) : (
                          <Field label="Physical time τ">
                            <input
                              aria-label="Physical time selection"
                              type="range"
                              min="0"
                              max={result.request.problem.grid.times.length - 1}
                              value={timeIndex}
                              onChange={(e) =>
                                setTimeIndex(Number(e.target.value))
                              }
                            />
                            <span>
                              {result.request.problem.grid.times[timeIndex]}
                            </span>
                          </Field>
                        )}
                        <Field label="Ensemble view">
                          <select
                            aria-label="Ensemble view"
                            value={sampleIndex}
                            onChange={(e) =>
                              setSampleIndex(Number(e.target.value))
                            }
                          >
                            <option value="-1">Ensemble mean</option>
                            <option value="-2">
                              Empirical standard deviation
                            </option>
                            {Array.from(
                              {
                                length: Number(
                                  result.request.inference.particles,
                                ),
                              },
                              (_, j) => (
                                <option key={j} value={j}>
                                  Sample {j + 1}
                                </option>
                              ),
                            )}
                          </select>
                        </Field>
                        {result.request.problem.grid.shape.length === 3 && (
                          <>
                            <Field label="Slice normal">
                              <select
                                aria-label="Slice normal"
                                value={sliceAxis}
                                onChange={(e) => {
                                  setSliceAxis(Number(e.target.value));
                                  setSliceIndex(0);
                                }}
                              >
                                {["x", "y", "z"].map((a, j) => (
                                  <option key={a} value={j}>
                                    {a} axis
                                  </option>
                                ))}
                              </select>
                            </Field>
                            <Field label="Slice index">
                              <input
                                aria-label="Slice index"
                                type="range"
                                min="0"
                                max={
                                  result.request.problem.grid.shape[sliceAxis] -
                                  1
                                }
                                value={sliceIndex}
                                onChange={(e) =>
                                  setSliceIndex(Number(e.target.value))
                                }
                              />
                              <span>
                                {sliceIndex} · coordinate{" "}
                                {slice?.slice_coordinate}
                              </span>
                            </Field>
                          </>
                        )}
                        <Button
                          variant="outline"
                          onClick={() =>
                            saveFile("configuration.json", result.request)
                          }
                        >
                          Export configuration
                        </Button>
                      </CardContent>
                    </Card>
                    {slice && (
                      <>
                        <p className="small-text">
                          {slice.axes.join("–")} orientation ·{" "}
                          {slice.physical_time === null
                            ? "static field"
                            : `physical τ=${slice.physical_time}`}
                          {slice.slice_coordinate !== null
                            ? ` · normal coordinate ${slice.slice_coordinate}`
                            : ""}{" "}
                          · {slice.statistic}
                        </p>
                        <div className="result-display-controls">
                          <label>
                            <input
                              type="checkbox"
                              aria-label="Show observation overlay"
                              checked={showObservations}
                              onChange={(e) =>
                                setShowObservations(e.target.checked)
                              }
                            />{" "}
                            Observation markers
                          </label>
                          <label>
                            <input
                              type="checkbox"
                              aria-label="Show evaluation reference and error"
                              disabled={!slice.reference}
                              checked={showReference && !!slice.reference}
                              onChange={(e) =>
                                setShowReference(e.target.checked)
                              }
                            />{" "}
                            Evaluation-only reference & error
                          </label>
                          <p>
                            Reconstruction and reference share a scale. Spread
                            and absolute error use their own labelled scales.
                          </p>
                        </div>
                        <details className="secondary-plots">
                          <summary>
                            Individual samples, empirical spread & saved input
                            plots
                          </summary>
                          <div className="plots coordinated-results">
                            <FieldPlot
                              slice={slice}
                              values={slice.values}
                              title={
                                sampleIndex === -2
                                  ? "Empirical spread"
                                  : "Reconstruction"
                              }
                              observations={showObservations}
                            />
                            {spreadSlice && sampleIndex !== -2 && (
                              <FieldPlot
                                slice={spreadSlice}
                                values={spreadSlice.values}
                                title="Empirical spread"
                              />
                            )}
                            <SavedInputPreview
                              jobId={job!.id}
                              field={
                                result.request.problem.fields[fieldIndex].name
                              }
                              time={timeIndex}
                              axis={sliceAxis}
                              index={sliceIndex}
                            />
                            {slice.reference && showReference && (
                              <FieldPlot
                                slice={slice}
                                values={slice.reference}
                                title="Reference"
                              />
                            )}
                            {slice.error && showReference && (
                              <FieldPlot
                                slice={slice}
                                values={slice.error}
                                title="Absolute error"
                              />
                            )}
                          </div>
                        </details>
                      </>
                    )}
                    {!result.reference_available && (
                      <Alert>
                        <AlertTitle>
                          No evaluation reference attached
                        </AlertTitle>
                        <AlertDescription>
                          Observation and physics diagnostics remain available.
                          Reconstruction error is unavailable.
                        </AlertDescription>
                      </Alert>
                    )}
                    <div className="result-outcome" role="status">
                      <strong>
                        {result.metrics.stop_reason === "FINISH_CRITERIA_MET"
                          ? "Configured physical / observation criteria met"
                          : result.metrics.stop_reason === "STOP_CANCELLED"
                            ? "Cancelled"
                            : result.metrics.stop_reason ===
                                "FAIL_INVALID_NUMERICS"
                              ? "Numerical failure"
                              : result.metrics.stop_reason ===
                                  "FINISH_BUDGET_EXHAUSTED"
                                ? "Execution finished: criteria unresolved / budget exhausted"
                                : "Generation finished; physical criteria not certified"}
                      </strong>
                      <p>
                        Finished execution is separate from reconstruction
                        accuracy. Inspect the recorded physical/observation
                        checks below.
                      </p>
                    </div>
                    <ResultCharts jobId={job!.id} />
                    <FKPanel metrics={result.metrics} />
                    <details className="expert-result">
                      <summary>
                        Physical verification, process status & actions
                      </summary>
                      <ProcessPanel metrics={result.metrics} />
                    </details>
                    <Card>
                      <CardHeader>
                        <CardTitle>Physical diagnostics</CardTitle>
                        <CardDescription>
                          Empirical spread is not automatically calibrated
                          uncertainty. Periodic boundaries are imposed by
                          representation.
                          {result.request.problem.family === "chemistry" &&
                            " Only the integrals of a+c and b+c are reaction invariants. Individual species and their unweighted sum need not conserve; sparse observations of one species do not guarantee unique hidden-species recovery."}
                        </CardDescription>
                      </CardHeader>
                      <CardContent>
                        <details>
                          <summary>All physical and reference metrics</summary>
                          <div className="metric-grid">
                            {[
                              "observation_rmse",
                              "pde_fd_rmse",
                              "pde_spectral_space_rmse",
                              ...(result.request.problem.family === "chemistry"
                                ? ["total_species_sum_max_change"]
                                : ["mass_max_drift"]),
                              "ensemble_rms_spread",
                              "relative_l2_unobserved_mean",
                              ...(result.request.problem.family === "chemistry"
                                ? [
                                    "minimum_concentration",
                                    "positivity_violations",
                                    "invariant_a_plus_c_max_drift",
                                    "invariant_b_plus_c_max_drift",
                                  ]
                                : ["field_u_mae"]),
                            ].map((k) => (
                              <div key={k}>
                                <small>{k.replaceAll("_", " ")}</small>
                                <strong>
                                  {typeof result.metrics[k] === "number"
                                    ? (result.metrics[k] as number).toPrecision(
                                        4,
                                      )
                                    : "Unavailable"}
                                </strong>
                              </div>
                            ))}
                          </div>
                        </details>
                        <FieldDiagnostics metrics={result.metrics} />
                        <EvaluationPanel metrics={result.metrics} />
                        {Object.entries(result.baselines ?? {}).map(
                          ([name, baseline]) => (
                            <details
                              key={name}
                              aria-label="Conventional inverse comparison"
                            >
                              <summary>Conventional inverse: {name}</summary>
                              <p>
                                Observation RMSE:{" "}
                                {typeof baseline.observation_rmse === "number"
                                  ? baseline.observation_rmse.toPrecision(4)
                                  : "Unavailable"}{" "}
                                ·{" "}
                                {result.request.problem.family === "darcy"
                                  ? "pressure is obtained from the numerical forward solve."
                                  : "conventional inverse reconstruction uses the stated truncated spectral basis."}
                              </p>
                              <FieldDiagnostics
                                metrics={baseline}
                                label={`Baseline ${name}: per-field reconstruction errors`}
                              />
                              <pre className="provenance">
                                {JSON.stringify(
                                  {
                                    cost: baseline.cost,
                                    stop_reason: baseline.stop_reason,
                                  },
                                  null,
                                  2,
                                )}
                              </pre>
                            </details>
                          ),
                        )}
                        <div className="toolbar">
                          <a
                            className="download-link"
                            href={`/api/jobs/${job!.id}/artifacts/samples.npz`}
                          >
                            Download numerical results
                          </a>
                          <a
                            className="download-link"
                            href={`/api/jobs/${job!.id}/artifacts/metrics.json`}
                          >
                            Download metrics
                          </a>
                          <a
                            className="download-link"
                            href={`/api/jobs/${job!.id}/experiment.zip`}
                          >
                            Export complete experiment
                          </a>
                          {referenceId && (
                            <details>
                              <summary>Comparison coordinate policy</summary>
                              <label>
                                Comparison policy
                                <select
                                  aria-label="Comparison policy"
                                  value={comparisonPolicy}
                                  onChange={(e) =>
                                    setComparisonPolicy(e.target.value)
                                  }
                                >
                                  <option value="native">
                                    Native identical grid
                                  </option>
                                  <option value="restrict_reference">
                                    Restrict fine reference
                                  </option>
                                </select>
                              </label>
                              <label>
                                <input
                                  type="checkbox"
                                  checked={sharedComparisonTimes}
                                  onChange={(e) =>
                                    setSharedComparisonTimes(e.target.checked)
                                  }
                                />{" "}
                                Exact shared saved times only
                              </label>
                              <p>
                                Periodic nodes coincide; cell centres use
                                conservative volume averages. No interpolation.
                                Different discretizations can contribute to
                                error.
                              </p>
                            </details>
                          )}
                          {referenceId && (
                            <Button
                              variant="outline"
                              onClick={() =>
                                void action(async () => {
                                  await api(`/jobs/${job!.id}/reference`, {
                                    reference_id: referenceId,
                                    policy: comparisonPolicy,
                                    shared_times_only: sharedComparisonTimes,
                                  });
                                  setResult(
                                    await api("/jobs/" + job!.id + "/result"),
                                  );
                                })
                              }
                            >
                              Compare
                            </Button>
                          )}
                          {result.metrics.evaluation ? (
                            <a
                              className="download-link"
                              href={`/api/jobs/${job!.id}/artifacts/comparison.json`}
                            >
                              Export comparison report
                            </a>
                          ) : null}
                          {referenceId && (
                            <Button
                              variant="outline"
                              onClick={() =>
                                void action(async () => {
                                  await api(`/jobs/${job!.id}/reference`, {
                                    reference_id: referenceId,
                                    policy: comparisonPolicy,
                                    shared_times_only: sharedComparisonTimes,
                                  });
                                  setResult(
                                    await api("/jobs/" + job!.id + "/result"),
                                  );
                                })
                              }
                            >
                              Attach synthetic reference for evaluation
                            </Button>
                          )}
                          {result.reference_available && (
                            <Button
                              variant="outline"
                              onClick={() =>
                                void action(async () => {
                                  await api(`/jobs/${job!.id}/reference`, {
                                    reference_id: null,
                                  });
                                  setResult(
                                    await api("/jobs/" + job!.id + "/result"),
                                  );
                                })
                              }
                            >
                              Remove evaluation reference
                            </Button>
                          )}
                        </div>
                        <Dialog>
                          <DialogTrigger asChild>
                            <Button variant="ghost">
                              View provenance & resolved settings
                            </Button>
                          </DialogTrigger>
                          <DialogContent>
                            <DialogHeader>
                              <DialogTitle>
                                Reconstruction provenance
                              </DialogTitle>
                              <DialogDescription>
                                Actual configuration and costs for this
                                completed job.
                              </DialogDescription>
                            </DialogHeader>
                            <pre className="provenance">
                              {JSON.stringify(
                                {
                                  job: result.job,
                                  settings: result.request,
                                  cost: result.metrics.cost,
                                  provenance: result.metrics.provenance,
                                },
                                null,
                                2,
                              )}
                            </pre>
                          </DialogContent>
                        </Dialog>
                      </CardContent>
                    </Card>
                  </div>
                ))}
              {section === "Runs" && (
                <div className="stack">
                  <RunComparison
                    jobs={jobs.filter(
                      (j) =>
                        j.summary?.mode !== "learned_observation_conditioning",
                    )}
                  />
                  <Card>
                    <CardHeader>
                      <CardTitle>Persistent local jobs</CardTitle>
                      <CardDescription>
                        Job records survive browser refresh. A server restart
                        marks unfinished jobs interrupted.
                      </CardDescription>
                    </CardHeader>
                    <CardContent>
                      <Button variant="outline" onClick={() => void loadJobs()}>
                        <RefreshCw size={15} />
                        Refresh
                      </Button>
                      <div className="observation-table">
                        <table>
                          <thead>
                            <tr>
                              <th>Run</th>
                              <th>Status</th>
                              <th>Seed</th>
                              <th>Model type</th>
                              <th>Stage</th>
                              <th>Open</th>
                            </tr>
                          </thead>
                          <tbody>
                            {jobs.map((j) => (
                              <tr key={j.id}>
                                <td>{j.id.slice(0, 10)}</td>
                                <td>
                                  <Badge variant="outline">{j.status}</Badge>
                                </td>
                                <td>{j.seed}</td>
                                <td>
                                  {j.summary?.mode ===
                                  "learned_observation_conditioning"
                                    ? "Coordinate heat"
                                    : "Fixed grid"}
                                </td>
                                <td>{j.stage}</td>
                                <td>
                                  <Button
                                    variant="ghost"
                                    onClick={() => reopen(j)}
                                  >
                                    Open run
                                  </Button>
                                </td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                      {!jobs.length && (
                        <p className="empty">
                          No jobs yet. Create measurements and validate a
                          reconstruction.
                        </p>
                      )}
                    </CardContent>
                  </Card>
                </div>
              )}
            </>
          )}
          <footer className="status-bar">
            <span className="status-dot" />
            <span>
              {busy
                ? "Validating / working"
                : running
                  ? `${job.stage} · ${job.completed_steps ?? 0} steps`
                  : `${section === "Results" && result ? result.request.observations.indices.length : rows.length} measurements · ${section === "Results" && result ? "completed run" : validation ? "request compatible" : "request not yet validated"}`}
            </span>
            <Tooltip>
              <TooltipTrigger asChild>
                <span tabIndex={0}>
                  CPU / optional MPS / CUDA · no hidden fallback
                </span>
              </TooltipTrigger>
              <TooltipContent>
                Unavailable backends are disabled. Device compatibility is
                rechecked by Python before launch.
              </TooltipContent>
            </Tooltip>
          </footer>
          <div className="next-section">
            {section === "Problem" && (
              <Button onClick={() => setSection("Observations")}>
                Continue to observations <ArrowRight size={15} />
              </Button>
            )}
            {section === "Observations" && (
              <Button onClick={() => setSection("Reconstruction")}>
                Configure reconstruction <ArrowRight size={15} />
              </Button>
            )}
          </div>
        </main>
      </div>
    </TooltipProvider>
  );
}
function Field({
  label,
  children,
}: {
  label: string;
  children: React.ReactNode;
}) {
  return (
    <div className="form-field">
      <Label>{label}</Label>
      {children}
    </div>
  );
}
