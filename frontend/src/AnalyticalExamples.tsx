import { useEffect, useState } from "react";
import type { Problem } from "./types";
import { api } from "./types";
import { Button } from "./components/ui/button";
type Example = {
  id: string;
  label: string;
  initial_expression: string;
  reference_expression: string;
  distribution: string;
  decay_note: string;
};
export function AnalyticalExamples({
  problem,
  ready,
  inputs,
  onInputs,
  onVerify,
  expanded = false,
}: {
  expanded?: boolean;
  problem: Problem;
  ready: boolean;
  inputs: string;
  onInputs: (value: string) => void;
  onVerify: () => Promise<unknown>;
}) {
  const [rows, setRows] = useState<Example[]>([]),
    [chosen, setChosen] = useState("product"),
    [verification, setVerification] = useState<unknown>(null),
    [error, setError] = useState("");
  const key = JSON.stringify(problem);
  useEffect(() => {
    let live = true;
    if (!ready || problem.family !== "heat") return;
    void api<Example[]>("/analytical-examples", { problem })
      .then((r) => {
        if (live) setRows(Array.isArray(r) ? r : []);
      })
      .catch((e) => {
        if (live) setError(String(e));
      });
    return () => {
      live = false;
    };
  }, [key, ready, problem]);
  if (problem.family !== "heat") return null;
  const selected = rows.find((r) => r.id === chosen);
  let value: Record<string, { expression?: string }> = {};
  try {
    value = JSON.parse(inputs);
  } catch {
    /* Advanced invalid JSON is preserved. */
  }
  function edit(name: string, text: string) {
    try {
      const next = JSON.parse(inputs);
      next[name] = { expression: text };
      onInputs(JSON.stringify(next, null, 2));
      setVerification(null);
    } catch {
      setError(
        "Correct the advanced reference-input JSON before editing expressions.",
      );
    }
  }
  return (
    <details open={expanded || undefined}>
      <summary>
        Analytical example · editable initial and complete reference
      </summary>
      <p>
        The unknown u trajectory is reconstructed from disclosed measurements.
        u(x,0) and complete u_ref(x,t) below are reference-only; D is known
        diffusivity. No initial field is disclosed to inference automatically.
      </p>
      <select
        aria-label="Analytical heat example"
        value={chosen}
        onChange={(e) => setChosen(e.target.value)}
      >
        {rows.map((r) => (
          <option key={r.id} value={r.id}>
            {r.label}
          </option>
        ))}
      </select>
      {selected && (
        <>
          <p>{selected.distribution}</p>
          <p>{selected.decay_note}</p>
          <pre>
            u0 = {selected.initial_expression}
            {"\n"}u_ref = {selected.reference_expression}
          </pre>
          <Button
            variant="outline"
            onClick={() => {
              onInputs(
                JSON.stringify(
                  {
                    initial_u: { expression: selected.initial_expression },
                    analytical_reference: {
                      expression: selected.reference_expression,
                    },
                  },
                  null,
                  2,
                ),
              );
              setVerification(null);
            }}
          >
            Replace reference inputs with reviewed example
          </Button>
        </>
      )}
      <label>
        Reference-only initial condition u(x,0)
        <textarea
          aria-label="Initial condition expression"
          value={value.initial_u?.expression ?? ""}
          onChange={(e) => edit("initial_u", e.target.value)}
        />
      </label>
      <label>
        Complete evaluation-only analytical reference u_ref(x,t)
        <textarea
          aria-label="Analytical reference expression"
          value={value.analytical_reference?.expression ?? ""}
          onChange={(e) => edit("analytical_reference", e.target.value)}
        />
      </label>
      <p>
        Use pi, physical x/y/z, t and D; sin/cos/exp and bounded arithmetic.
        Simple Unicode and LaTeX function notation is normalized on
        verification. Times and domain remain the current reviewed layout.
        Candidates must pass PDE, initial and periodic value/gradient checks;
        finite numerical verification is not a general proof.
      </p>
      <Button
        variant="outline"
        disabled={!ready || !value.analytical_reference?.expression}
        onClick={() =>
          void onVerify()
            .then((v) => {
              setVerification(v);
              setError("");
            })
            .catch((e) => setError(String(e)))
        }
      >
        Verify analytical candidate
      </Button>
      {verification != null && (
        <pre aria-label="Analytical verification">
          {JSON.stringify(verification, null, 2)}
        </pre>
      )}
      {error && <p role="alert">{error}</p>}
    </details>
  );
}
