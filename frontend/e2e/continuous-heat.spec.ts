import { test, expect } from "@playwright/test";
import type { Page } from "@playwright/test";
import fs from "node:fs";
import path from "node:path";
const evidence = path.resolve("../outputs/continuous-time-20261028");
async function time(page: Page, t: number) {
  await page
    .getByLabel("Physical time", { exact: true })
    .evaluate((node, value) => {
      const el = node as HTMLInputElement;
      Object.getOwnPropertyDescriptor(
        HTMLInputElement.prototype,
        "value",
      )!.set!.call(el, String(value));
      el.dispatchEvent(new Event("input", { bubbles: true }));
    }, t);
  await expect(page.getByTestId("physical-time")).toHaveText(
    `t = ${t.toFixed(3)}`,
  );
  await expect(page.locator(".heat-stage")).toHaveAttribute(
    "data-evaluated-time",
    String(t),
  );
}
async function take(page: Page) {
  const response = page.waitForResponse(
    (r) =>
      r.url().endsWith("/playground/measurements") &&
      r.request().method() === "POST",
  );
  await page
    .getByRole("button", { name: "Take measurements", exact: true })
    .click();
  const result = await response;
  expect(result.ok()).toBeTruthy();
  return result.json();
}
test("continuous time and real random/even/focused/manual designs with stale guards", async ({
  page,
}) => {
  fs.mkdirSync(evidence, { recursive: true });
  await page.goto("/");
  await expect(
    page.getByRole("button", { name: "Play", exact: true }),
  ).toBeEnabled();
  await expect(
    page.getByText("Time interval: 0 to 1", { exact: true }),
  ).toBeVisible();
  await time(page, 0.237);
  await page.getByLabel("Number of measurements", { exact: true }).fill("25");
  const random = await take(page);
  expect(random.rows).toHaveLength(25);
  expect(
    random.rows.every((r: { time: number }) => r.time === 0.237),
  ).toBeTruthy();
  await page.getByLabel("Noise (σ)", { exact: true }).fill(".02");
  const noisy = await take(page);
  expect(noisy.rows.map((r: { x: number; y: number }) => [r.x, r.y])).toEqual(
    random.rows.map((r: { x: number; y: number }) => [r.x, r.y]),
  );
  await page
    .getByRole("button", { name: "Evenly spaced", exact: true })
    .click();
  const even = await take(page);
  expect(even.rows).toHaveLength(25);
  expect(
    new Set(even.rows.map((r: { x: number; y: number }) => `${r.x},${r.y}`))
      .size,
  ).toBe(25);
  await page
    .getByRole("button", { name: "Two hot spots", exact: true })
    .click();
  await expect(
    page.getByRole("button", { name: "Play", exact: true }),
  ).toBeEnabled();
  await time(page, 0);
  await page.getByRole("button", { name: "Heat-focused", exact: true }).click();
  const focused = await take(page);
  expect(focused.placement.retained_regions.length).toBeGreaterThanOrEqual(2);
  expect(focused.placement.background_count).toBe(7);
  await page.screenshot({
    path: path.join(evidence, "heat-focused.png"),
    fullPage: true,
  });
  await page.getByLabel("Focus spread", { exact: true }).evaluate((node) => {
    Object.getOwnPropertyDescriptor(
      HTMLInputElement.prototype,
      "value",
    )!.set!.call(node, ".12");
    node.dispatchEvent(new Event("input", { bubbles: true }));
  });
  const spread = await take(page);
  expect(spread.placement.spread_fraction_min_length).toBe(0.12);
  await page.getByRole("button", { name: "Click points", exact: true }).click();
  const canvas = page.getByRole("img", { name: "Heat field", exact: true });
  const b = (await canvas.boundingBox())!;
  await canvas.click({ position: { x: b.width * 0.25, y: b.height * 0.75 } });
  await expect(
    page.getByText("1 sensor points · click/tap to toggle", { exact: true }),
  ).toBeVisible();
  await canvas.click({ position: { x: b.width * 0.25, y: b.height * 0.75 } });
  await expect(
    page.getByText("0 sensor points · click/tap to toggle", { exact: true }),
  ).toBeVisible();
  for (const [x, y] of [
    [0.2, 0.2],
    [0.5, 0.6],
    [0.8, 0.3],
  ])
    await canvas.click({ position: { x: b.width * x, y: b.height * (1 - y) } });
  const manual = await take(page);
  expect(manual.rows).toHaveLength(3);
  expect(manual.rows[0].x).toBeCloseTo(0.2, 2);
  expect(manual.rows[0].y).toBeCloseTo(0.2, 2);
  await page.screenshot({
    path: path.join(evidence, "manual-sensors.png"),
    fullPage: true,
  });
  // Delay actual numerical responses: this checks UI races without mocking a solver.
  let intercepted = false;
  await page.route("**/api/playground/measurements", async (route) => {
    const response = await route.fetch();
    intercepted = true;
    await new Promise((r) => setTimeout(r, 700));
    await route.fulfill({ response });
  });
  await page.getByRole("button", { name: "Random", exact: true }).click();
  await time(page, 0.237);
  await page
    .getByRole("button", { name: "Take measurements", exact: true })
    .click();
  await expect.poll(() => intercepted).toBeTruthy();
  await time(page, 0.5);
  await page.waitForTimeout(800);
  await expect(page.getByTestId("measurement-summary")).not.toContainText(
    "t = 0.237",
  );
  await page.unroute("**/api/playground/measurements");
  // Frame count is a display choice, and cannot rewrite committed measurements.
  await page.locator(".playground-details>summary").click();
  await page.getByText("Display & export frames", { exact: true }).click();
  await page.getByLabel("Animation frame count", { exact: true }).fill("11");
  await expect(page.getByTestId("measurement-summary")).toContainText(
    "3 measurements",
  );
  let heldField = false,
    latestReference = "";
  await page.route("**/api/playground/reference", async (route) => {
    const response = await route.fetch();
    const body = route.request().postDataJSON();
    if (body.preset === "cosine") {
      heldField = true;
      await new Promise((resolve) => setTimeout(resolve, 700));
    } else latestReference = (await response.json()).id;
    await route.fulfill({ response });
  });
  await page.getByRole("button", { name: "Cosine waves", exact: true }).click();
  await expect.poll(() => heldField).toBeTruthy();
  await page.getByRole("button", { name: "Hot spot", exact: true }).click();
  await expect.poll(() => latestReference).not.toBe("");
  await expect(page.locator(".heat-stage")).toHaveAttribute(
    "data-reference-id",
    latestReference,
  );
  await page.waitForTimeout(800);
  await expect(page.locator(".heat-stage")).toHaveAttribute(
    "data-reference-id",
    latestReference,
  );
  await page.unroute("**/api/playground/reference");
  fs.writeFileSync(
    path.join(evidence, "browser-placement.json"),
    JSON.stringify({ random, noisy, even, focused, spread, manual }, null, 2),
  );
});

test("real continuous checkpoint, exact-time fixed-latent decoding and legacy warning", async ({
  page,
}) => {
  await page.goto("/");
  await expect(
    page.getByRole("button", { name: "Play", exact: true }),
  ).toBeEnabled();
  await page
    .getByLabel("Import heat model", { exact: true })
    .setInputFiles(path.join(evidence, "local/pipeline/export/coordinate.pt"));
  await expect(page.getByLabel("Heat model", { exact: true })).toBeVisible();
  await page.getByLabel("Barely trained", { exact: false }).check();
  await time(page, 0.237);
  await page.getByLabel("Number of measurements", { exact: true }).fill("25");
  await take(page);
  let generations = 0;
  page.on("request", (r) => {
    if (r.url().endsWith("/coordinate/reconstruct")) {
      generations++;
      const p = r.postDataJSON();
      expect(p.rows).toHaveLength(25);
      expect(
        p.rows.every((row: { time: number }) => row.time === 0.237),
      ).toBeTruthy();
      expect(p.require_continuous).toBe(true);
      expect(p.reference).toBeUndefined();
      expect(p.initial).toBeUndefined();
      expect(p.problem.known_fields).toEqual({});
    }
  });
  await page.getByRole("button", { name: "Run model", exact: true }).click();
  await expect(page.getByTestId("comparison")).toBeVisible({ timeout: 60000 });
  await time(page, 0.5);
  const response = page.waitForResponse(
    (r) => r.url().includes("/results/") && r.url().endsWith("/at?time=0.237"),
  );
  await time(page, 0.237);
  const decoded = await (await response).json();
  expect(decoded.fixed_latent).toBeTruthy();
  await time(page, 0.5);
  const again = page.waitForResponse(
    (r) => r.url().includes("/results/") && r.url().endsWith("/at?time=0.237"),
  );
  await time(page, 0.237);
  expect((await (await again).json()).frame).toEqual(decoded.frame);
  expect(generations).toBe(1);
  await page.screenshot({
    path: path.join(evidence, "continuous-reconstruction.png"),
    fullPage: true,
  });
  const legacy = path.resolve(
    "../outputs/colab-training-20261027/notebook-run/export/coordinate.pt",
  );
  await page
    .getByLabel("Import heat model", { exact: true })
    .setInputFiles(legacy);
  const select = page.getByLabel("Heat model", { exact: true });
  await expect(
    select
      .locator("option")
      .filter({ hasText: "unavailable for this experiment" }),
  ).toHaveCount(1);
  await select.selectOption(
    (await select
      .locator("option")
      .filter({ hasText: "unavailable for this experiment" })
      .getAttribute("value")) as string,
  );
  await expect(
    page.getByText(/Legacy snapshot checkpoint is not continuously trained/),
  ).toBeVisible();
});
