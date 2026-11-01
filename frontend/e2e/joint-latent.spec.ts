import { test, expect } from "@playwright/test";
import fs from "node:fs";
import { createHash } from "node:crypto";
import path from "node:path";
const out = path.resolve("../outputs/joint-latent-20261030");
const checkpoint = path.join(out, "verified-cpu/export/coordinate.pt");
test("real joint model import, known/unknown D, coherent time and metrics", async ({
  page,
}) => {
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(e.message));
  await page.goto("/");
  await expect(
    page.getByRole("button", { name: "Play", exact: true }),
  ).toBeEnabled();
  await page
    .getByLabel("Import heat model", { exact: true })
    .setInputFiles(checkpoint);
  await expect(page.getByLabel("Heat model", { exact: true })).toBeVisible();
  const importedId=createHash("sha256").update(fs.readFileSync(checkpoint)).digest("hex").slice(0,32);
  await page.getByLabel("Heat model",{exact:true}).selectOption(importedId);
  await page.getByLabel("Barely trained", { exact: false }).check();
  const measurement = page.waitForResponse((r) =>
    r.url().endsWith("/playground/measurements"),
  );
  await page
    .getByRole("button", { name: "Take measurements", exact: true })
    .click();
  expect((await measurement).ok()).toBeTruthy();
  const evidence: unknown[] = [];
  let generations = 0;
  page.on("request", (r) => {
    if (r.url().endsWith("/coordinate/reconstruct")) generations++;
  });
  for (const infer of [false, true]) {
    await page
      .getByLabel("Infer diffusivity jointly", { exact: true })
      .setChecked(infer);
    await page.getByLabel("Barely trained", { exact: false }).check();
    const request = page.waitForRequest((r) =>
      r.url().endsWith("/coordinate/reconstruct"),
    );
    await page.getByRole("button", { name: "Run model", exact: true }).click();
    const submitted = (await request).postDataJSON();
    expect(submitted.model_id).toBe(importedId);
    expect(submitted.problem.known_fields || {}).toEqual({});
    expect(submitted.problem.initial_condition).toBe("unknown");
    expect(submitted.problem.known_parameters).toEqual(
      infer ? {} : { diffusivity: 0.07 },
    );
    expect(submitted.problem.unknown_parameters).toEqual(
      infer ? ["diffusivity"] : [],
    );
    await expect(page.getByTestId("comparison")).toBeVisible({
      timeout: 60000,
    });
    await expect(page.getByText(/Diffusivity samples:/)).toBeVisible();
    await page
      .getByLabel("Error metric", { exact: true })
      .selectOption("region_balanced_mse");
    await page.getByLabel("Error map", { exact: true }).selectOption("region");
    await page.getByLabel("Show detected regions", { exact: true }).check();
    await expect(page.locator(".result-metrics b").first()).not.toHaveText("—");
    const beforeTime = generations;
    await page.getByLabel("Physical time", { exact: true }).evaluate((node) => {
      const el = node as HTMLInputElement;
      Object.getOwnPropertyDescriptor(
        HTMLInputElement.prototype,
        "value",
      )!.set!.call(el, "0.5");
      el.dispatchEvent(new Event("input", { bubbles: true }));
    });
    await expect(page.locator(".heat-stage")).toHaveAttribute(
      "data-evaluated-time",
      "0.5",
    );
    const exactFrame = page.waitForResponse(
      (r) =>
        r.url().includes("/results/") && r.url().endsWith("/at?time=0.237"),
    );
    await page.getByLabel("Physical time", { exact: true }).evaluate((node) => {
      const el = node as HTMLInputElement;
      Object.getOwnPropertyDescriptor(
        HTMLInputElement.prototype,
        "value",
      )!.set!.call(el, "0.237");
      el.dispatchEvent(new Event("input", { bubbles: true }));
    });
    expect((await exactFrame).ok()).toBeTruthy();
    await expect(page.locator(".heat-stage")).toHaveAttribute(
      "data-evaluated-time",
      "0.237",
    );
    await expect(page.locator(".result-metrics b").first()).not.toHaveText("—");
    expect(generations).toBe(beforeTime);
    await page.screenshot({
      path: path.join(
        out,
        infer ? "browser-unknown-D.png" : "browser-known-D.png",
      ),
      fullPage: true,
    });
    evidence.push({
      infer,
      submitted,
      parameterSummary: await page
        .getByText(/Diffusivity samples:/)
        .textContent(),
    });
  }
  expect(errors).toEqual([]);
  fs.writeFileSync(
    path.join(out, "browser-inference.json"),
    JSON.stringify({ evidence, errors }, null, 2),
  );
});
test("training UI new defaults and real bounded managed training", async ({
  page,
}) => {
  await page.goto("/");
  await page
    .getByRole("button", { name: "Train model", exact: true })
    .first()
    .click();
  await expect(
    page.getByLabel("Field reconstruction loss", { exact: true }),
  ).toHaveValue("region_balanced_mse");
  await expect(
    page.getByLabel("Feature balance: α", { exact: true }),
  ).toHaveValue("0.5");
  await page
    .getByLabel("Training budget", { exact: true })
    .selectOption("pipeline-check");
  await page.getByLabel("Training device", { exact: true }).selectOption("cpu");
  await page
    .getByText("Advanced settings and architecture", { exact: true })
    .click();
  for (const [label, value] of [
    ["Representation updates", "2"],
    ["Flow training updates", "2"],
    ["Representation width", "24"],
    ["Latent tokens", "8"],
    ["Encoder points", "32"],
    ["Training query points", "32"],
    ["Validate and checkpoint every updates", "1"],
  ])
    await page.getByLabel(label, { exact: true }).fill(value);
  await page.screenshot({
    path: path.join(out, "browser-training.png"),
    fullPage: true,
  });
  const start = page.waitForResponse((r) =>
    r.url().endsWith("/training/start"),
  );
  await page
    .getByRole("button", { name: "Train model", exact: true })
    .last()
    .click();
  const response = await start;
  expect(response.ok()).toBeTruthy();
  const run = await response.json();
  await expect
    .poll(
      async () =>
        (await (await page.request.get(`/api/training/runs/${run.id}`)).json())
          .state,
      { timeout: 60000 },
    )
    .toBe("completed");
  const result = await (
    await page.request.get(`/api/training/runs/${run.id}`)
  ).json();
  fs.writeFileSync(
    path.join(out, "browser-training.json"),
    JSON.stringify(result, null, 2),
  );
});
