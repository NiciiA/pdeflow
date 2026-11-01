import { test, expect } from "@playwright/test";
import fs from "node:fs";
import { createHash } from "node:crypto";
import path from "node:path";
const evidence = path.resolve("../outputs/field-losses-20261029");
test("new training UI selects real alpha=.5 loss and runs bounded encoder/decoder training", async ({
  page,
}) => {
  await page.goto("/");
  await page
    .getByRole("button", { name: "Train model", exact: true })
    .first()
    .click();
  const loss = page.getByLabel("Field reconstruction loss", { exact: true });
  await expect(loss).toHaveValue("mse");
  await loss.selectOption("region_balanced_mse");
  await expect(
    page.getByLabel("Feature balance: α", { exact: true }),
  ).toHaveValue("0.5");
  await page
    .getByLabel("Training budget", { exact: true })
    .selectOption("pipeline-check");
  await page.getByLabel("Training device", { exact: true }).selectOption("cpu");
  await page.getByText("Advanced region analysis", { exact: true }).click();
  await page.getByLabel("Analysis grid per axis", { exact: true }).fill("8");
  await page.getByLabel("Supervised time groups", { exact: true }).fill("2");
  await page
    .getByLabel("Additional region queries / solution", { exact: true })
    .fill("32");
  await page
    .getByText("Advanced settings and architecture", { exact: true })
    .click();
  for (const [label, value] of [
    ["Representation updates", "2"],
    ["Flow training updates", "2"],
    ["Representation width", "8"],
    ["Latent tokens", "2"],
    ["Attention heads", "2"],
    ["Encoder points", "16"],
    ["Training query points", "32"],
    ["Batch size", "1"],
    ["Validate and checkpoint every updates", "1"],
  ])
    await page.getByLabel(label, { exact: true }).fill(value);
  await page
    .getByText("Preview regions on training examples", { exact: true })
    .click();
  const preview = page.waitForResponse((r) =>
    r.url().endsWith("/training/loss-preview"),
  );
  await page
    .getByRole("button", { name: "Preview training regions", exact: true })
    .click();
  const data = await (await preview).json();
  expect(data.split).toBe("train");
  expect(data.examples).toHaveLength(6);
  await expect(page.locator(".loss-preview-grid canvas")).toHaveCount(6);
  await page
    .locator(".field-loss-controls")
    .screenshot({ path: path.join(evidence, "training-loss-preview.png") });
  const starting = page.waitForResponse((r) =>
    r.url().endsWith("/training/start"),
  );
  await page
    .getByRole("button", { name: "Train model", exact: true })
    .last()
    .click();
  const response = await starting;
  expect(response.ok()).toBeTruthy();
  const run = await response.json();
  expect(run.plan.config.field_loss.type).toBe("region_balanced_mse");
  expect(run.plan.config.field_loss.alpha).toBe(0.5);
  await expect
    .poll(
      async () =>
        (await (await page.request.get(`/api/training/runs/${run.id}`)).json())
          .state,
      { timeout: 60000 },
    )
    .toBe("completed");
  const final = await (
    await page.request.get(`/api/training/runs/${run.id}`)
  ).json();
  expect(final.plan.config.field_loss).toEqual(run.plan.config.field_loss);
  fs.writeFileSync(
    path.join(evidence, "browser-training.json"),
    JSON.stringify({ run: final, preview: data }, null, 2),
  );
});

test("actual comparison: metrics and overlays reuse predictions; maps are labelled contributions", async ({
  page,
}) => {
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(e.message));
  await page.goto("/");
  await expect(
    page.getByRole("button", { name: "Play", exact: true }),
  ).toBeEnabled();
  await page
    .getByRole("button", { name: "Two hot spots", exact: true })
    .click();
  await expect(
    page.getByRole("button", { name: "Play", exact: true }),
  ).toBeEnabled();
  const importedId = createHash("sha256")
    .update(
      fs.readFileSync(
        path.join(evidence, "notebook/notebook-run/export/coordinate.pt"),
      ),
    )
    .digest("hex")
    .slice(0, 32);
  await page
    .getByLabel("Import heat model", { exact: true })
    .setInputFiles(
      path.join(evidence, "notebook/notebook-run/export/coordinate.pt"),
    );
  await expect(page.getByLabel("Heat model", { exact: true })).toBeVisible();
  await expect(
    page
      .getByLabel("Heat model", { exact: true })
      .locator(`option[value="${importedId}"]`),
  ).toHaveCount(1);
  await page.getByLabel("Heat model", { exact: true }).selectOption(importedId);
  await page.getByLabel("Barely trained", { exact: false }).check();
  const measurement = page.waitForResponse((r) =>
    r.url().endsWith("/playground/measurements"),
  );
  await page
    .getByRole("button", { name: "Take measurements", exact: true })
    .click();
  expect((await measurement).ok()).toBeTruthy();
  const observations = await page
    .getByTestId("measurement-summary")
    .innerText();
  let generations = 0,
    metricCalls = 0;
  page.on("request", (r) => {
    if (r.url().endsWith("/coordinate/reconstruct")) {
      generations++;
      expect(r.postDataJSON().model_id).toBe(importedId);
    }
    if (r.url().endsWith("/playground/error-metrics")) metricCalls++;
  });
  const metrics = page.waitForResponse(
    (r) =>
      r.url().endsWith("/playground/error-metrics") &&
      r.request().method() === "POST",
  );
  await page.getByRole("button", { name: "Run model", exact: true }).click();
  await expect(page.getByTestId("comparison")).toBeVisible({ timeout: 60000 });
  const report = await (await metrics).json();
  expect(report.mse).toBeGreaterThan(0);
  expect(report.region_count).toBeGreaterThanOrEqual(2);
  const count = metricCalls;
  for (const [id, title] of [
    ["mse", "MSE"],
    ["mae", "MAE"],
    ["relative_l2", "Relative L2"],
    ["weighted_mse", "Weighted MSE"],
    ["region_balanced_mse", "Region-balanced MSE"],
    ["rmse", "RMSE"],
  ]) {
    await page.getByLabel("Error metric", { exact: true }).selectOption(id);
    await expect(page.locator(".result-metrics")).toContainText(title);
    await expect(page.locator(".result-metrics b").first()).toHaveText(
      report[id].toPrecision(4),
    );
  }
  await page
    .getByLabel("Error metric", { exact: true })
    .selectOption("region_balanced_mse");
  await page.getByLabel("Error map", { exact: true }).selectOption("region");
  await page.getByLabel("Show detected regions", { exact: true }).check();
  await expect(
    page.getByRole("img", {
      name: "Region-balanced MSE contribution",
      exact: true,
    }),
  ).toBeVisible();
  expect(generations).toBe(1);
  expect(metricCalls).toBe(count);
  await expect(page.getByTestId("measurement-summary")).toHaveText(
    observations,
  );
  await page.getByText("Error metric details", { exact: true }).click();
  await page.screenshot({
    path: path.join(evidence, "region-overlay.png"),
    fullPage: true,
  });
  await page.getByLabel("Error metric", { exact: true }).selectOption("rmse");
  await page.getByLabel("Error map", { exact: true }).selectOption("absolute");
  await expect(
    page.getByRole("img", {
      name: "Absolute error |model − reference|",
      exact: true,
    }),
  ).toBeVisible();
  await page.screenshot({
    path: path.join(evidence, "metric-selection.png"),
    fullPage: true,
  });
  expect(errors).toEqual([]);
  fs.writeFileSync(
    path.join(evidence, "browser-metrics.json"),
    JSON.stringify(
      { report, importedId, generations, metricCalls, errors },
      null,
      2,
    ),
  );
  await page.locator(".playground-details>summary").click();
  await page
    .getByLabel("Measurements only (no reference comparison)", { exact: true })
    .check();
  await expect(page.getByLabel("Error metric", { exact: true })).toHaveCount(0);
});
