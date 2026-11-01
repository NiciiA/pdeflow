import { test, expect } from "@playwright/test";
import type { Page } from "@playwright/test";
import fs from "node:fs";
import path from "node:path";
const evidence = path.resolve("../outputs/heat-playground-20261027");
const checkpoint = path.resolve(
  "../outputs/coordinate-20261027/workspace/coordinate/heat-flow/coordinate.pt",
);
const workspace = "/private/tmp/pdeflow-heat-playground-20261027";
async function ready(page: Page) {
  await expect(
    page.getByRole("button", { name: "Play", exact: true }),
  ).toBeEnabled();
  await expect(
    page.getByText("Updating simulation", { exact: true }),
  ).toHaveCount(0);
}
async function open(page: Page) {
  await page.goto("/");
  await ready(page);
}
async function advanced(page: Page) {
  await page.locator(".playground-details>summary").click();
}
test.describe.configure({ mode: "serial" });
test("no-model play, exact cosine, drawing, seeded random, formula and state", async ({
  page,
}) => {
  await open(page);
  await expect(
    page.getByText("No trained model yet. Simulation remains available."),
  ).toBeVisible();
  await page.screenshot({
    path: path.join(evidence, "main-playground.png"),
    fullPage: true,
  });
  await page.getByRole("button", { name: "Play", exact: true }).click();
  await expect(page.getByTestId("physical-time")).not.toHaveText("t = 0.000");
  await page.getByRole("button", { name: "Pause", exact: true }).click();
  const ref = page.waitForResponse(
    (r) =>
      r.url().endsWith("/playground/reference") &&
      r.request().postDataJSON().D === 0.1,
  );
  await page.getByLabel("Diffusivity", { exact: true }).fill("0.1");
  const rr = await (await ref).json();
  expect(rr.exact_max_error).toBeLessThan(1e-13);
  await ready(page);
  await page.getByLabel("Amplitude", { exact: true }).fill("0.6");
  await page.getByLabel("Waves in x", { exact: true }).fill("2");
  await ready(page);
  await page.getByLabel("Reference view").selectOption("exact");
  await expect(
    page.getByRole("img", { name: "Heat field", exact: true }),
  ).toBeVisible();
  await page.getByLabel("Reference view").selectOption("difference");
  await expect(
    page.getByRole("img", { name: "Absolute numerical − exact difference" }),
  ).toBeVisible();
  await page.getByRole("button", { name: "Restart simulation" }).click();
  await expect(page.getByTestId("physical-time")).toHaveText("t = 0.000");
  await page
    .getByRole("button", { name: "Draw your own", exact: true })
    .click();
  await ready(page);
  const canvas = page.getByRole("img", { name: "Heat field", exact: true });
  const box = (await canvas.boundingBox())!;
  const painted = page.waitForResponse(
    (r) =>
      r.url().endsWith("/playground/reference") &&
      r.request().postDataJSON().preset === "draw",
  );
  await page.mouse.move(box.x + 3, box.y + box.height * 0.5);
  await page.mouse.down();
  await page.mouse.move(box.x + box.width * 0.3, box.y + box.height * 0.6, {
    steps: 10,
  });
  await page.mouse.up();
  const dr = await (await painted).json();
  expect(dr.initial.flat().some((x: number) => x > 1)).toBeTruthy();
  await ready(page);
  const cleared = page.waitForResponse((r) =>
    r.url().endsWith("/playground/reference"),
  );
  await page.getByRole("button", { name: "Clear field", exact: true }).click();
  expect(
    (await (await cleared).json()).initial.flat().every((x: number) => x === 0),
  ).toBeTruthy();
  await ready(page);
  await page.getByRole("button", { name: "Undo stroke" }).click();
  await ready(page);
  const random = page.waitForResponse((r) =>
    r.url().endsWith("/playground/reference"),
  );
  await page
    .getByRole("button", { name: "Smooth random", exact: true })
    .click();
  const a = await (await random).json();
  await ready(page);
  await page.getByRole("button", { name: "New random field" }).click();
  await ready(page);
  const reproduced = page.waitForResponse((r) =>
    r.url().endsWith("/playground/reference"),
  );
  await page.getByLabel("Random field seed").fill("73");
  expect((await (await reproduced).json()).initial).toEqual(a.initial);
  await ready(page);
  await page.getByRole("button", { name: "Formula", exact: true }).click();
  await ready(page);
  await page.getByLabel("Starting field formula").fill("unknown+1");
  await expect(page.getByRole("alert")).toContainText("unknown symbol");
  await expect(canvas).toBeVisible();
  await page.getByLabel("Starting field formula").fill("sin(2*pi*x)");
  await ready(page);
  // Delayed real responses prove stale response rejection; no numerical API is mocked.
  let delayed = true;
  await page.route("**/api/playground/reference", async (route) => {
    const response = await route.fetch();
    if (delayed) {
      delayed = false;
      await new Promise((r) => setTimeout(r, 650));
    }
    await route.fulfill({ response });
  });
  await page.getByLabel("Diffusivity", { exact: true }).fill(".09");
  await page.waitForRequest((r) => r.url().endsWith("/playground/reference"));
  await page.getByLabel("Diffusivity", { exact: true }).fill(".08");
  await ready(page);
  await page.waitForTimeout(800);
  await expect(page.getByLabel("Diffusivity", { exact: true })).toHaveValue(
    /0?\.08/,
  );
  await page.unroute("**/api/playground/reference");
  await page.getByRole("button", { name: "Cosine waves", exact: true }).click();
  await ready(page);
  await page.getByLabel("Number of measurements").fill("0");
  await page
    .getByRole("button", { name: "Take measurements", exact: true })
    .click();
  await expect(page.getByRole("alert")).toContainText("between 1 and 2000");
  await page.getByLabel("Number of measurements").fill("25");
  await page.getByLabel("Noise (σ)").fill(".03");
  const sampled = page.waitForResponse((r) =>
    r.url().endsWith("/playground/measurements"),
  );
  await page
    .getByRole("button", { name: "Take measurements", exact: true })
    .click();
  const samples = await (await sampled).json();
  expect(samples.rows).toHaveLength(25);
  expect(
    JSON.parse(
      fs.readFileSync(
        path.join(
          workspace,
          "workbench/data",
          samples.reference_id,
          "problem.json",
        ),
        "utf8",
      ),
    ).known_parameters.diffusivity,
  ).toBe(0.08);
  expect(
    samples.rows.every((r: { sigma: number }) => r.sigma === 0.03),
  ).toBeTruthy();
  await expect(page.getByTestId("measurement-summary")).toContainText(
    "25 measurements at t = 0.000",
  );
  await page.getByLabel("Physical time", { exact: true }).fill("15");
  await expect(page.getByTestId("measurement-summary")).toContainText(
    "t = 0.000",
  );
  const next = page.waitForResponse((r) =>
    r.url().endsWith("/playground/measurements"),
  );
  await page.getByRole("button", { name: "Resample measurements" }).click();
  const b = await (await next).json();
  expect(b.seed).toBe(samples.seed + 1);
  expect(b.rows).not.toEqual(samples.rows);
  await page
    .getByRole("button", { name: "Train model", exact: true })
    .first()
    .click();
  await expect(page.getByLabel("Training budget")).toHaveValue("development");
  await expect(page.getByLabel("Training model")).toHaveCount(0);
  await page.getByRole("button", { name: "Back to playground" }).click();
  await ready(page);
});

test("existing coordinate checkpoint: real inverse, isolation, comparison, device and D checks", async ({
  page,
}) => {
  await open(page);
  await page
    .getByLabel("Import heat model", { exact: true })
    .setInputFiles(checkpoint);
  await expect(page.getByLabel("Heat model", { exact: true })).toBeVisible();
  await page.getByLabel("Barely trained", { exact: false }).check();
  await page.getByLabel("Number of measurements").fill("25");
  await page
    .getByRole("button", { name: "Take measurements", exact: true })
    .click();
  await expect(page.getByTestId("measurement-summary")).toContainText(
    "25 measurements",
  );
  await advanced(page);
  await page.getByLabel("Inference device").selectOption("cuda");
  await page.getByRole("button", { name: "Run model", exact: true }).click();
  await expect(page.getByRole("alert")).toContainText(
    "CUDA requested but unavailable",
  );
  await page.getByLabel("Inference device").selectOption("cpu");
  await page.getByLabel("Model output detail").selectOption("16");
  let requestBody: Record<string, unknown> | undefined;
  page.on("request", (r) => {
    if (r.url().endsWith("/coordinate/reconstruct"))
      requestBody = r.postDataJSON();
  });
  await page.getByRole("button", { name: "Run model", exact: true }).click();
  await expect(page.getByTestId("comparison")).toBeVisible({ timeout: 60000 });
  expect(requestBody).toBeDefined();
  expect(Object.keys(requestBody!)).not.toContain("reference");
  expect(Object.keys(requestBody!)).not.toContain("initial");
  expect(
    (requestBody!.problem as { known_fields: object }).known_fields,
  ).toEqual({});
  await expect(
    page.locator(".result-metrics>span").filter({ hasText: "Field RMSE" }),
  ).toBeVisible();
  await page.locator(".playground-details>summary").click();
  await page.screenshot({
    path: path.join(evidence, "model-comparison.png"),
    fullPage: true,
  });
  await page.getByLabel("Physical time", { exact: true }).fill("15");
  await expect(page.getByTestId("measurement-summary")).toContainText(
    "t = 0.000",
  );
  await page.getByLabel("Diffusivity", { exact: true }).fill(".5");
  await ready(page);
  await expect(
    page.getByText("This model supports D between 0.03 and 0.12."),
  ).toBeVisible();
  await expect(page.getByTestId("comparison")).toHaveCount(0);
  await page.getByLabel("Diffusivity", { exact: true }).fill(".07");
  await ready(page);
});

test("measurements-only inference omits truth; active deletion blocked and cancellation works", async ({
  page,
}) => {
  await open(page);
  await page.getByLabel("Barely trained", { exact: false }).check();
  await advanced(page);
  await page.getByLabel("Import measurements", { exact: true }).setInputFiles({
    name: "observations.json",
    mimeType: "application/json",
    buffer: Buffer.from(
      JSON.stringify([
        { field: "u", x: 0.2, y: 0.4, time: 0.05, value: 0.3, sigma: 0.03 },
        { field: "u", x: 0.7, y: 0.1, time: 0.1, value: -0.2, sigma: 0.03 },
      ]),
    ),
  });
  await page.getByLabel("Model output detail").selectOption("16");
  await page.getByRole("button", { name: "Run model", exact: true }).click();
  await expect(page.getByTestId("comparison")).toBeVisible({ timeout: 60000 });
  await expect(
    page.getByRole("img", { name: "Reference", exact: true }),
  ).toHaveCount(0);
  await expect(
    page.locator(".result-metrics>span").filter({ hasText: "Field RMSE" }),
  ).toHaveCount(0);
  await expect(
    page.getByRole("img", { name: "Model reconstruction", exact: true }),
  ).toBeVisible();
  await page.getByLabel("Heun steps").fill("512");
  await page.getByRole("button", { name: "Run model", exact: true }).click();
  await expect(
    page.getByRole("button", { name: "Cancel reconstruction" }),
  ).toBeVisible();
  await page.getByRole("button", { name: "Storage", exact: true }).click();
  await expect(page.getByText("Active · protected").first()).toBeVisible();
  await page
    .getByRole("button", { name: "Delete all model artifacts…" })
    .click();
  await expect(page.getByRole("alert")).toContainText(
    "reconstruction is active",
  );
  await page.getByRole("button", { name: "Back to playground" }).click();
  await page.getByRole("button", { name: "Cancel reconstruction" }).click();
  await expect(
    page.getByText("Reconstruction cancelled. You can retry."),
  ).toBeVisible();
});

test("disposable storage: preview, cancel, permanent scope, selection clearing, zero-model play", async ({
  page,
  request,
}) => {
  // Only the explicitly isolated temporary service is permitted for this destructive test.
  let inv = await (await request.get("/api/playground/storage")).json();
  if (!inv.models.length) {
    await request.post("/api/playground/import", {
      headers: { "Content-Type": "application/octet-stream" },
      data: fs.readFileSync(checkpoint),
    });
    inv = await (await request.get("/api/playground/storage")).json();
  }
  expect(inv.root).toBe(workspace);
  expect(inv.models.length).toBeGreaterThan(0);
  fs.writeFileSync(path.join(workspace, "protected-personal.txt"), "keep");
  const m = inv.models[0];
  await request.post("/api/training/use", {
    data: { id: m.model_ids[0], problem: null, adopt_training_problem: true },
  });
  await open(page);
  await page.getByRole("button", { name: "Storage", exact: true }).click();
  await expect(page.getByText("Total managed")).toBeVisible();
  await page.locator(".storage-list input[type=checkbox]").first().check();
  await page.getByRole("button", { name: "Preview selected deletion" }).click();
  await expect(
    page.getByRole("heading", { name: "Review permanent deletion" }),
  ).toBeVisible();
  await page.getByRole("button", { name: "Cancel deletion" }).click();
  expect(fs.existsSync(path.join(workspace, m.files[0]))).toBeTruthy();
  await page
    .getByRole("button", { name: "Delete all model artifacts…" })
    .click();
  await expect(
    page.getByText("Generated data remains. Results/logs remain."),
  ).toBeVisible();
  await page
    .getByLabel("Deletion confirmation")
    .fill("DELETE ALL MODEL ARTIFACTS");
  await page
    .getByRole("button", { name: "Permanently delete", exact: true })
    .click();
  await expect(
    page.getByText(
      "No managed model weights. The heat simulation remains available.",
    ),
  ).toBeVisible({ timeout: 30000 });
  expect(
    fs.readFileSync(path.join(workspace, "protected-personal.txt"), "utf8"),
  ).toBe("keep");
  expect(
    fs.existsSync(path.join(workspace, "selected-model.json")),
  ).toBeFalsy();
  await page.getByRole("button", { name: "Back to playground" }).click();
  await expect(
    page.getByText("No trained model yet. Simulation remains available."),
  ).toBeVisible();
  await page.getByRole("button", { name: "Play", exact: true }).click();
  await expect(page.getByTestId("physical-time")).not.toHaveText("t = 0.000");
});
