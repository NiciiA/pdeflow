import { test, expect } from "@playwright/test";
import fs from "node:fs";
import path from "node:path";
import { createHash } from "node:crypto";
const out = path.resolve("../outputs/measurement-times-20261101");
const cp = path.resolve(
  "../outputs/operator-wave-20261030/final-cpu/flow/coordinate.pt",
);
const button = (page: import("@playwright/test").Page) =>
  page.getByRole("button", { name: "Take measurements", exact: true });
test("real random and arbitrary-time sensing, fixed committed timestamps and exact-time inference", async ({
  page,
}) => {
  test.setTimeout(180000);
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(e.message));
  const evidence: unknown[] = [];
  await page.goto("/");
  await expect(button(page)).toBeEnabled();
  await page.getByLabel("Import heat model", { exact: true }).setInputFiles(cp);
  const id = createHash("sha256")
    .update(fs.readFileSync(cp))
    .digest("hex")
    .slice(0, 32);
  await expect(page.getByLabel("Heat model", { exact: true })).toBeVisible();
  await page.getByLabel("Heat model", { exact: true }).selectOption(id);
  await page.getByLabel("Number of measurements", { exact: true }).fill("7");
  for (const family of ["heat", "wave"]) {
    if (family === "wave") {
      await page.getByLabel("Equation", { exact: true }).selectOption("wave");
      await page
        .getByRole("button", { name: "Travelling wave", exact: true })
        .click();
      await expect(button(page)).toBeEnabled();
      await page
        .getByLabel("Measurement quantity", { exact: true })
        .selectOption("v");
      await page.getByLabel("Measurement-time policy").selectOption("chosen");
      await page
        .getByLabel("Chosen physical times", { exact: true })
        .fill("0.02, 0.2, 0.47, 0.83");
    } else
      await expect(page.getByLabel("Measurement-time policy")).toHaveValue(
        "continuous",
      );
    const sampled = page.waitForResponse((r) =>
      r.url().endsWith("/playground/measurements"),
    );
    await button(page).click();
    const response = await sampled;
    expect(response.ok()).toBeTruthy();
    const record = await response.json();
    expect(record.rows).toHaveLength(7);
    const times: number[] = record.rows.map((r: { time: number }) => r.time);
    if (family === "heat") {
      expect(new Set(times).size).toBe(7);
      expect(Math.min(...times)).toBeLessThan(0.15);
      expect(Math.max(...times)).toBeGreaterThan(0.85);
    } else expect([...new Set(times)].sort()).toEqual([0.02, 0.2, 0.47, 0.83]);
    await page
      .getByText("Inspect sensor timestamps and values", { exact: true })
      .click();
    await expect(page.getByTestId("sensor-time-0")).toBeVisible();
    let regeneration = 0;
    const counter = (r: import("@playwright/test").Request) => {
      if (r.url().endsWith("/playground/measurements")) regeneration++;
    };
    page.on("request", counter);
    await page.getByLabel("Physical time", { exact: true }).evaluate((node) => {
      const e = node as HTMLInputElement;
      Object.getOwnPropertyDescriptor(
        HTMLInputElement.prototype,
        "value",
      )!.set!.call(e, "0.61");
      e.dispatchEvent(new Event("input", { bubbles: true }));
    });
    await expect(page.getByTestId("sensor-time-0")).toBeVisible();
    await page.getByLabel("Barely trained", { exact: false }).check();
    const launched = page.waitForResponse((r) =>
      r.url().endsWith("/coordinate/reconstruct"),
    );
    const submitted = page.waitForRequest((r) =>
      r.url().endsWith("/coordinate/reconstruct"),
    );
    await page.getByRole("button", { name: "Run model", exact: true }).click();
    const payload = (await submitted).postDataJSON();
    expect(payload.rows).toEqual(record.rows);
    await expect(page.getByTestId("comparison")).toBeVisible({
      timeout: 60000,
    });
    expect(regeneration).toBe(0);
    page.off("request", counter);
    const started = await (await launched).json();
    const actual = await (
      await page.request.get(`/api/playground/results/${started.id}`)
    ).json();
    expect(
      actual.metrics.observation_consistency.sensor_coordinates_and_times,
    ).toEqual(
      record.rows.map((r: { x: number; y: number; time: number }) => [
        r.x,
        r.y,
        r.time,
      ]),
    );
    expect(actual.metrics.sensor_diagnostics.physical_times).toEqual(
      [...new Set(times)].sort((a, b) => a - b),
    );
    evidence.push({ family, record, payload, metrics: actual.metrics });
    await page.screenshot({
      path: path.join(out, `browser-${family}-times.png`),
      fullPage: true,
    });
  }
  expect(errors).toEqual([]);
  fs.writeFileSync(
    path.join(out, "browser-evidence.json"),
    JSON.stringify(evidence, null, 2),
  );
});
