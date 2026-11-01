import { useEffect, useState } from "react";
import { api } from "./types";
import type { Slice } from "./types";
import { FieldPlot } from "./FieldPlot";
import { Button } from "./components/ui/button";
type Frame = {
  generation_position: number;
  step: number;
  particle_slots: number[];
  parents_at_previous_snapshot: number[];
  endpoint_available: boolean;
};
type Metadata = {
  frames: Frame[];
  field_indices: number[];
  field_names?: string[];
  time_indices: number[];
  native_grid: { shape: number[]; times: number[]; temporal_kind: string };
  display_shape: number[];
  lineage: string;
  diagnostic_model_evaluations: number;
};
export function GenerationPlayer({ jobId }: { jobId: string }) {
  const [meta, setMeta] = useState<Metadata | null>(null),
    [frame, setFrame] = useState(0),
    [particle, setParticle] = useState(0),
    [capturedField, setCapturedField] = useState(0),
    [capturedTime, setCapturedTime] = useState(0),
    [axis, setAxis] = useState(2),
    [index, setIndex] = useState(0),
    [kind, setKind] = useState("current_physical"),
    [slice, setSlice] = useState<Slice | null>(null),
    [playing, setPlaying] = useState(false),
    [error, setError] = useState("");
  useEffect(() => {
    let live = true;
    void api<Metadata>(`/jobs/${jobId}/generation`)
      .then((v) => {
        if (live) {
          setMeta(v);
          setFrame(0);
          setCapturedField(v.field_indices[0]);
          setCapturedTime(v.time_indices[0]);
          setError("");
        }
      })
      .catch(() => {
        if (live)
          setError(
            "Generation snapshots were not recorded for this saved run. Enable snapshot capture before a new reconstruction.",
          );
      });
    return () => {
      live = false;
    };
  }, [jobId]);
  useEffect(() => {
    if (!meta) return;
    let live = true;
    void api<Slice>(
      `/jobs/${jobId}/generation?frame=${frame}&particle=${particle}&kind=${kind}&field_index=${capturedField}&time_index=${capturedTime}&axis=${axis}&index=${index}`,
    )
      .then((v) => {
        if (live) setSlice(v);
      })
      .catch((e) => {
        if (live) setError(String(e));
      });
    return () => {
      live = false;
    };
  }, [
    jobId,
    meta,
    frame,
    particle,
    kind,
    capturedField,
    capturedTime,
    axis,
    index,
  ]);
  useEffect(() => {
    if (!playing || !meta) return;
    const timer = setInterval(
      () =>
        setFrame((f) => {
          if (f >= meta.frames.length - 1) {
            setPlaying(false);
            return f;
          }
          return f + 1;
        }),
      650,
    );
    return () => clearInterval(timer);
  }, [playing, meta]);
  const f = meta?.frames[frame],
    parent = f?.parents_at_previous_snapshot[particle],
    jump = frame > 0 && parent !== particle;
  return (
    <section className="generation-player">
      <h3>Noise → candidate field</h3>
      <p>
        Generation progress s is distinct from physical time τ. These are
        captured sampler states, not a simulated physical evolution or an
        interpolation between images.
      </p>
      {error && <p role="status">{error}</p>}
      {meta && f && (
        <>
          <div className="playback">
            <Button
              aria-label="Previous generation snapshot"
              onClick={() => {
                setPlaying(false);
                setFrame((n) => Math.max(0, n - 1));
              }}
            >
              ←
            </Button>
            <Button onClick={() => setPlaying((v) => !v)}>
              {playing ? "Pause generation" : "Play generation"}
            </Button>
            <Button
              aria-label="Next generation snapshot"
              onClick={() => {
                setPlaying(false);
                setFrame((n) => Math.min(meta.frames.length - 1, n + 1));
              }}
            >
              →
            </Button>
            <label>
              Generation progress s
              <input
                aria-label="Generation progress s"
                type="range"
                min={0}
                max={meta.frames.length - 1}
                value={frame}
                onChange={(e) => {
                  setPlaying(false);
                  setFrame(Number(e.target.value));
                }}
              />
            </label>
            <strong>s={f.generation_position.toFixed(4)}</strong>
          </div>
          <div className="inspection-toolbar">
            <label>
              Captured field
              <select
                aria-label="Generation field"
                value={capturedField}
                onChange={(e) => setCapturedField(Number(e.target.value))}
              >
                {meta.field_indices.map((c, i) => (
                  <option key={c} value={c}>
                    {meta.field_names?.[i] ?? `Field ${c}`}
                  </option>
                ))}
              </select>
            </label>
            {meta.native_grid.temporal_kind !== "static" && (
              <label>
                Captured physical time τ
                <select
                  aria-label="Generation physical time"
                  value={capturedTime}
                  onChange={(e) => setCapturedTime(Number(e.target.value))}
                >
                  {meta.time_indices.map((t) => (
                    <option key={t} value={t}>
                      {meta.native_grid.times[t]}
                    </option>
                  ))}
                </select>
              </label>
            )}
            {meta.native_grid.shape.length === 3 && (
              <>
                <label>
                  Snapshot slice normal
                  <select
                    aria-label="Generation slice normal"
                    value={axis}
                    onChange={(e) => {
                      setAxis(Number(e.target.value));
                      setIndex(0);
                    }}
                  >
                    {["x", "y", "z"].map((a, i) => (
                      <option key={a} value={i}>
                        {a}
                      </option>
                    ))}
                  </select>
                </label>
                <label>
                  Captured slice index
                  <input
                    aria-label="Generation slice index"
                    type="range"
                    min={0}
                    max={meta.display_shape[axis] - 1}
                    value={index}
                    onChange={(e) => setIndex(Number(e.target.value))}
                  />
                </label>
              </>
            )}
            <label>
              Captured particle
              <select
                aria-label="Generation particle"
                value={particle}
                onChange={(e) => setParticle(Number(e.target.value))}
              >
                {f.particle_slots.map((p) => (
                  <option key={p} value={p}>
                    {p}
                  </option>
                ))}
              </select>
            </label>
            <label>
              Generative quantity
              <select
                aria-label="Generative quantity"
                value={kind}
                onChange={(e) => setKind(e.target.value)}
              >
                <option value="current_physical">
                  Current noisy/generated state · decoded
                </option>
                <option value="current_model">
                  Raw current model coordinates
                </option>
                <option value="endpoint_physical">Estimated final field</option>
              </select>
            </label>
            <Button variant="outline" onClick={() => setFrame(0)}>
              Initial noise
            </Button>
            <Button
              variant="outline"
              onClick={() => setFrame(meta.frames.length - 1)}
            >
              Last captured state
            </Button>
          </div>
          <p>
            Captured field index {capturedField} ·{" "}
            {meta.native_grid.temporal_kind === "static"
              ? "static field"
              : `fixed physical τ=${meta.native_grid.times[capturedTime]}`}{" "}
            · native {meta.native_grid.shape.join("×")}, display{" "}
            {meta.display_shape.join("×")}. {meta.diagnostic_model_evaluations}{" "}
            passive endpoint model calls are accounted separately.
          </p>
          {jump && (
            <p className="lineage-notice">
              Ancestry transition: slot {particle} descends from previous
              captured slot {parent}.{" "}
              {parent! >= f.particle_slots.length
                ? "Parent was outside the bounded recorded subset; the path has a gap."
                : "Follow the parent slot to inspect the preceding state."}
              <Button
                variant="ghost"
                disabled={parent! >= f.particle_slots.length}
                onClick={() => {
                  setParticle(parent!);
                  setFrame((n) => n - 1);
                  setPlaying(false);
                }}
              >
                Follow recorded parent
              </Button>
            </p>
          )}
          {kind === "endpoint_physical" && !f.endpoint_available ? (
            <p>Endpoint estimate unavailable at this captured boundary.</p>
          ) : (
            slice && (
              <FieldPlot
                slice={slice}
                values={slice.values}
                title={
                  kind === "endpoint_physical"
                    ? "Estimated final field"
                    : kind === "current_model"
                      ? "Raw current model state"
                      : "Current noisy/generated field"
                }
              />
            )
          )}
          <small>
            {meta.lineage} The returned reconstruction is shown in Spatial
            inspection.
          </small>
        </>
      )}
    </section>
  );
}
