import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { saveFile } from "./types";
import type { Model } from "./types";
const quote = (value: string) => "'" + value.replaceAll("'", "'\\''") + "'";
export function TrainingDataCard({ model }: { model: Model | undefined }) {
  const [mode, setMode] = useState("online"),
    [evaluation, setEvaluation] = useState("outputs/my-training/evaluation"),
    [output, setOutput] = useState("outputs/my-training/training");
  if (!model)
    return (
      <p>
        No compatible checkpoint selected. Implemented generator support is
        shown separately above; training is launched explicitly with the CLI.
      </p>
    );
  const provider = {
    distribution_version: "physical_v1",
    solver_rtol: 1e-10,
    solver_max_iterations: 4000,
    calibration_count: 8,
    validation_count: 2,
    test_count: 2,
    offline_count: 32,
    capacity: 8,
    reuse_limit: 2,
    max_attempts: 2,
    workers: 0,
    solver_threads: 1,
    ...model.config.provider,
    mode,
  };
  const configuration = { ...model.config, provider };
  const configName = "training-config.json",
    data = output + "-data";
  const cli = ".venv/bin/python -m pdeflow.cli ";
  const commands = [
    "#!/bin/sh",
    "set -e",
    "# Run from the repository root after saving training-config.json there.",
    cli +
      "prepare-evaluation --config " +
      quote(configName) +
      " --output " +
      quote(evaluation),
    ...(mode === "offline"
      ? [
          cli +
            "prepare-training --config " +
            quote(configName) +
            " --output " +
            quote(data),
        ]
      : []),
    cli +
      "train --config " +
      quote(configName) +
      " --evaluation " +
      quote(evaluation) +
      " --output " +
      quote(output) +
      (mode === "offline" ? " --data " + quote(data) : ""),
    "# Resume with unchanged source/config/manifest:",
    cli +
      "train --config " +
      quote(configName) +
      " --evaluation " +
      quote(evaluation) +
      " --output " +
      quote(output) +
      (mode === "offline" ? " --data " + quote(data) : "") +
      " --resume " +
      quote(output + "/checkpoint.pt"),
  ].join("\n");
  return (
    <Card>
      <CardHeader>
        <CardTitle>Training data configuration</CardTitle>
      </CardHeader>
      <CardContent className="stack">
        <p>
          Checkpoint data mode:{" "}
          {model.training.data_mode ??
            String(model.config.provider?.mode ?? "offline")}
          . Fixed evaluation identity:{" "}
          {model.training.evaluation_identity ??
            "not recorded (legacy offline)"}
          .
        </p>
        <label>
          Exported training mode
          <select
            aria-label="Exported training mode"
            value={mode}
            onChange={(e) => setMode(e.target.value)}
          >
            <option value="offline">Offline saved references</option>
            <option value="online">Online fresh references</option>
            <option value="hybrid">Hybrid bounded replay</option>
          </select>
        </label>
        <p>
          Reference generation is supervised numerical solving. Exporting a mode
          or opening a preset never starts training. Producer: workers=0, no
          prefetch; hybrid capacity={String(provider.capacity)}, maximum
          presentations per cached problem={String(provider.reuse_limit)}.
        </p>
        <details>
          <summary>Physical sampler distribution and cache settings</summary>
          <pre className="provenance">
            {JSON.stringify(
              {
                data: model.config.data,
                chemistry: model.config.chemistry,
                provider,
              },
              null,
              2,
            )}
          </pre>
        </details>
        <label>
          Fixed evaluation manifest directory
          <Input
            aria-label="Fixed evaluation manifest directory"
            value={evaluation}
            onChange={(e) => setEvaluation(e.target.value)}
          />
        </label>
        <label>
          Training output directory
          <Input
            aria-label="Training output directory"
            value={output}
            onChange={(e) => setOutput(e.target.value)}
          />
        </label>
        <p>
          Save the configuration in the repository root. Preparation commands
          require fresh directories; resume reuses the same immutable evaluation
          manifest.
        </p>
        <Button
          variant="outline"
          onClick={() => saveFile(configName, configuration)}
        >
          Export training configuration
        </Button>
        <Button
          variant="outline"
          onClick={() => saveFile("training-commands.sh", commands)}
        >
          Export training commands
        </Button>
      </CardContent>
    </Card>
  );
}
