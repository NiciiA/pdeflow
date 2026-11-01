import { test, expect } from "@playwright/test";
import fs from "node:fs";
import path from "node:path";
import { createHash } from "node:crypto";
const out = path.resolve("../outputs/operator-wave-20261030");
const checkpoint = path.join(out, "final-cpu/flow/coordinate.pt");
test("real shared graph-conditioned heat/wave known and unknown reconstruction", async ({
  page,
}) => {
  test.setTimeout(180000);
  const errors: string[] = [];
  const evidence: unknown[] = [];
  page.on("pageerror", (e) => errors.push(e.message));
  await page.goto("/");
  await expect(
    page.getByRole("button", { name: "Play", exact: true }),
  ).toBeEnabled();
  await page
    .getByLabel("Import heat model", { exact: true })
    .setInputFiles(checkpoint);
  const id = createHash("sha256")
    .update(fs.readFileSync(checkpoint))
    .digest("hex")
    .slice(0, 32);
  let generations = 0;
  page.on("request", (r) => {
    if (r.url().endsWith("/coordinate/reconstruct")) generations++;
  });
  for (const family of ["heat", "wave"]) {
    await page.getByLabel("Equation", { exact: true }).selectOption(family);
    if (family === "wave")
      await page
        .getByRole("button", { name: "Travelling wave", exact: true })
        .click();
    await expect(
      page.getByRole("button", { name: "Take measurements", exact: true }),
    ).toBeEnabled();
    await expect(page.getByLabel("Heat model", { exact: true })).toBeVisible();
    await page.getByLabel("Heat model", { exact: true }).selectOption(id);
    for (const infer of [false, true]) {
      await page
        .getByLabel(
          family === "wave"
            ? "Infer wave speed c jointly"
            : "Infer diffusivity D jointly",
          { exact: true },
        )
        .setChecked(infer);
      if (family === "wave")
        await page
          .getByLabel("Measurement quantity", { exact: true })
          .selectOption(infer ? "v" : "u");
      const measurement = page.waitForResponse((r) =>
        r.url().endsWith("/playground/measurements"),
      );
      await page
        .getByRole("button", { name: "Take measurements", exact: true })
        .click();
      const measured = await measurement;
      expect(measured.ok()).toBeTruthy();
      const sampling = await measured.json();
      await page.getByLabel("Barely trained", { exact: false }).check();
      const request = page.waitForRequest((r) =>
        r.url().endsWith("/coordinate/reconstruct"),
      );
      await page
        .getByRole("button", { name: "Run model", exact: true })
        .click();
      const submitted = (await request).postDataJSON();
      expect(submitted.model_id).toBe(id);
      expect(submitted.problem.family).toBe(family);
      expect(submitted.problem.operator_spec.family).toBe(family);
      expect(submitted.problem.known_fields || {}).toEqual({});
      expect(
        submitted.rows.every(
          (r: { field: string }) =>
            r.field === (family === "wave" && infer ? "v" : "u"),
        ),
      ).toBeTruthy();
      expect(submitted.problem.unknown_parameters).toEqual(
        infer ? [family === "heat" ? "diffusivity" : "wave_speed"] : [],
      );
      if (infer) expect(submitted.problem.known_parameters).toEqual({});
      await expect(page.getByTestId("comparison")).toBeVisible({
        timeout: 60000,
      });
      if (family === "wave")
        await page
          .getByLabel("Display quantity", { exact: true })
          .selectOption(infer ? "v" : "u");
      await page
        .getByLabel("Error metric", { exact: true })
        .selectOption("region_balanced_mse");
      await page
        .getByLabel("Error map", { exact: true })
        .selectOption("region");
      await page.getByLabel("Show detected regions", { exact: true }).check();
      await expect(page.locator(".result-metrics b").first()).not.toHaveText(
        "—",
      );
      const before = generations;
      await page
        .getByLabel("Physical time", { exact: true })
        .evaluate((node) => {
          const e = node as HTMLInputElement;
          Object.getOwnPropertyDescriptor(
            HTMLInputElement.prototype,
            "value",
          )!.set!.call(e, "0.5");
          e.dispatchEvent(new Event("input", { bubbles: true }));
        });
      await expect(page.locator(".heat-stage")).toHaveAttribute(
        "data-evaluated-time",
        "0.5",
      );
      const timeResponse = page.waitForResponse(
        (r) =>
          r.url().includes("/results/") && r.url().includes("/at?time=0.237"),
      );
      await page
        .getByLabel("Physical time", { exact: true })
        .evaluate((node) => {
          const e = node as HTMLInputElement;
          Object.getOwnPropertyDescriptor(
            HTMLInputElement.prototype,
            "value",
          )!.set!.call(e, "0.237");
          e.dispatchEvent(new Event("input", { bubbles: true }));
        });
      expect((await timeResponse).ok()).toBeTruthy();
      await expect(page.locator(".heat-stage")).toHaveAttribute(
        "data-evaluated-time",
        "0.237",
      );
      expect(generations).toBe(before);
      await page
        .getByLabel("Error metric", { exact: true })
        .selectOption("mae");
      expect(generations).toBe(before);
      await page.screenshot({
        path: path.join(
          out,
          `browser-${family}-${infer ? "unknown" : "known"}.png`,
        ),
        fullPage: true,
      });
      evidence.push({ family, infer, sampling, submitted });
    }
  }
  expect(errors).toEqual([]);
  fs.writeFileSync(
    path.join(out, "browser-four-modes.json"),
    JSON.stringify({ evidence, generations, errors }, null, 2),
  );
});
test("wave references, independent velocity drawing and all measurement placements", async ({
  page,
  request,
}) => {
  await page.goto("/");
  await page.getByLabel("Equation", { exact: true }).selectOption("wave");
  for (const preset of [
    "Standing mode",
    "Travelling wave",
    "Displacement pulse",
    "Velocity-only pulse",
    "Signed pulses",
    "Random u and v",
  ]) {
    const response = page.waitForResponse((r) =>
      r.url().endsWith("/playground/reference"),
    );
    await page.getByRole("button", { name: preset, exact: true }).click();
    const r = await response;
    expect(r.ok()).toBeTruthy();
    const ref = await r.json();
    expect(ref.problem.fields.map((f: { name: string }) => f.name)).toEqual([
      "u",
      "v",
    ]);
    expect(ref.velocity_frames.length).toBe(31);
  }
  await page
    .getByRole("button", { name: "Draw your own", exact: true })
    .click();
  await expect(page.getByLabel("Draw quantity", { exact: true })).toBeVisible();
  await page.getByLabel("Draw quantity", { exact: true }).selectOption("v");
  await expect(
    page.getByLabel("Display quantity", { exact: true }),
  ).toHaveValue("v");
  const beforeResponse = await request.post("/api/playground/reference", {
    data: {
      family: "wave",
      preset: "draw",
      initial: Array.from({ length: 64 }, () => Array(64).fill(0.2)),
      velocity_initial: Array.from({ length: 64 }, () => Array(64).fill(0)),
    },
  });
  expect(beforeResponse.ok()).toBeTruthy();
  const before = await beforeResponse.json();
  await expect(
    page.getByRole("button", { name: "Play", exact: true }),
  ).toBeEnabled();
  const canvas = page.locator("canvas.paintable").first();
  const box = await canvas.boundingBox();
  expect(box).toBeTruthy();
  const updated = page.waitForResponse(
    (r) =>
      r.url().endsWith("/playground/reference") &&
      !!r.request().postDataJSON()?.velocity_initial,
  );
  await page.mouse.move(box!.x + box!.width * 0.4, box!.y + box!.height * 0.4);
  await page.mouse.down();
  await page.mouse.move(box!.x + box!.width * 0.5, box!.y + box!.height * 0.5, {
    steps: 6,
  });
  await page.mouse.up();
  const drawn = await (await updated).json();
  expect(
    drawn.initial_velocity.flat().some((v: number) => v !== 0),
  ).toBeTruthy();
  // Every placement evaluates the selected physical quantity at actual continuous times.
  for (const placement of ["random", "even", "heat", "manual"]) {
    const data = {
      reference_id: before.id,
      quantity: "v",
      count: placement === "manual" ? 2 : 5,
      noise: 0,
      seed: 91,
      time: 0.237,
      placement,
      positions: [
        [0.1, 0.2],
        [0.4, 0.5],
      ],
      pattern: placement === "manual" ? "manual" : "current",
    };
    const result = await request.post("/api/playground/measurements", { data });
    expect(result.ok()).toBeTruthy();
    const rows = (await result.json()).rows;
    expect(
      rows.every(
        (r: { field: string; time: number }) =>
          r.field === "v" && r.time === 0.237,
      ),
    ).toBeTruthy();
  }
  await page.screenshot({
    path: path.join(out, "browser-wave-velocity-drawing.png"),
    fullPage: true,
  });
});
