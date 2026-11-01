import { test, expect } from "@playwright/test";
import type { Page, Locator } from "@playwright/test";
import path from "node:path";
import fs from "node:fs";
const evidence = path.resolve("../outputs/spatial-profiles-20261028");
type Field = number[][];
async function ready(page: Page) {
  await expect(
    page.getByRole("button", { name: "Play", exact: true }),
  ).toBeEnabled();
  await expect(page.locator(".heat-stage")).toHaveAttribute(
    "data-evaluated-time",
    "0",
  );
}
async function preset(page: Page, name: string) {
  const r = page.waitForResponse(
    (r) =>
      r.url().endsWith("/playground/reference") &&
      r.request().method() === "POST",
  );
  await page.getByRole("button", { name, exact: true }).click();
  const body = await (await r).json();
  await expect(page.locator(".heat-stage")).toHaveAttribute(
    "data-reference-id",
    body.id,
  );
  await ready(page);
  return body;
}
async function at(page: Page, t: number) {
  const r = page.waitForResponse(
    (r) => r.url().includes("/reference/") && r.url().endsWith(`/at?time=${t}`),
  );
  await page.getByLabel("Physical time", { exact: true }).evaluate((el, t) => {
    Object.getOwnPropertyDescriptor(
      HTMLInputElement.prototype,
      "value",
    )!.set!.call(el, String(t));
    el.dispatchEvent(new Event("input", { bubbles: true }));
  }, t);
  const body = await (await r).json();
  await expect(page.locator(".heat-stage")).toHaveAttribute(
    "data-evaluated-time",
    String(t),
  );
  return body.frame as Field;
}
async function profileCheck(
  figure: Locator,
  field: Field,
  lengths = [1, 1],
  magnitude = false,
) {
  for (const axis of ["x", "y"] as const) {
    const plot = figure.locator(`.profile-${axis}`),
      n = axis === "x" ? field.length : field[0].length;
    const scale = JSON.parse(
      (await plot.getAttribute("data-scale"))!,
    ) as number[];
    const values =
      axis === "x"
        ? field.map(
            (row) =>
              (row.reduce((s, v) => s + (magnitude ? Math.abs(v) : v), 0) *
                lengths[1]) /
              field[0].length,
          )
        : field[0].map(
            (_, j) =>
              (field.reduce(
                (s, row) => s + (magnitude ? Math.abs(row[j]) : row[j]),
                0,
              ) *
                lengths[0]) /
              field.length,
          );
    const points = (await plot.locator(".profile-curve").getAttribute("d"))!
      .match(/[-+\d.e]+,[-+\d.e]+/g)!
      .map((v) => v.split(",").map(Number));
    expect(points).toHaveLength(n + 1);
    values.forEach((v, i) => {
      expect(points[i][axis === "x" ? 0 : 1]).toBeCloseTo(
        (axis === "x" ? i / n : 1 - i / n) * 1000,
        5,
      );
      const expected = (1 - (v - scale[0]) / (scale[1] - scale[0])) * 1000;
      // Near-zero cosine integrals differ at FP roundoff between initial and Fourier-at-time arrays.
      // 0.1 SVG units is < .06 screen pixels; finite-amplitude cases retain the stricter check.
      expect(Math.abs(points[i][axis === "x" ? 1 : 0] - expected)).toBeLessThan(
        scale[1] - scale[0] < 1e-10 ? 0.1 : 0.0001,
      );
    });
  }
}
async function alignment(figure: Locator, ratio = 1) {
  const c = (await figure.locator("canvas").boundingBox())!,
    x = (await figure.locator(".profile-x svg").boundingBox())!,
    y = (await figure.locator(".profile-y svg").boundingBox())!;
  expect(x.x).toBeCloseTo(c.x, 1);
  expect(x.width).toBeCloseTo(c.width, 1);
  expect(y.y).toBeCloseTo(c.y, 1);
  expect(y.height).toBeCloseTo(c.height, 1);
  expect(y.x + y.width).toBeLessThan(c.x);
  expect(x.y).toBeGreaterThan(c.y + c.height - 1);
  expect(c.width / c.height).toBeCloseTo(ratio, 2);
}

test("real fields: aligned integrals, time, drawing, signed/magnitude and responsive layout", async ({
  page,
}) => {
  fs.mkdirSync(evidence, { recursive: true });
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(e.message));
  await page.goto("/");
  await ready(page);
  const figure = page.locator(".heat-stage .heat-figure").first();
  const spot = await preset(page, "Hot spot");
  await profileCheck(figure, spot.frames[0]);
  await alignment(figure);
  // Pixel centres and marginal maxima share physical node positions (not half-cell offsets).
  const pixelPeak = await figure
    .locator("canvas")
    .evaluate((canvas: HTMLCanvasElement) => {
      const ctx = canvas.getContext("2d")!,
        im = ctx.getImageData(0, 0, canvas.width, canvas.height);
      let best = Infinity,
        sx = 0,
        sy = 0,
        count = 0;
      for (let y = 0; y < im.height; y++)
        for (let x = 0; x < im.width; x++) {
          const i = (y * im.width + x) * 4,
            distance =
              (im.data[i] - 211) ** 2 +
              (im.data[i + 1] - 69) ** 2 +
              (im.data[i + 2] - 63) ** 2;
          if (distance < best) {
            best = distance;
            sx = 0;
            sy = 0;
            count = 0;
          }
          if (distance === best) {
            sx += x + 0.5;
            sy += y + 0.5;
            count++;
          }
        }
      return [sx / count / im.width, 1 - sy / count / im.height];
    });
  expect(pixelPeak[0]).toBeCloseTo(0.34375, 2);
  expect(pixelPeak[1]).toBeCloseTo(0.5, 2);
  await figure.screenshot({ path: path.join(evidence, "hot-spot.png") });
  const scales = await figure.locator(".profile-x").getAttribute("data-scale");
  const firstPath = await figure
    .locator(".profile-x .profile-curve")
    .getAttribute("d");
  const later = await at(page, 0.237);
  await profileCheck(figure, later);
  expect(await figure.locator(".profile-x").getAttribute("data-scale")).toBe(
    scales,
  );
  expect(
    await figure.locator(".profile-x .profile-curve").getAttribute("d"),
  ).not.toBe(firstPath);
  // Playback uses current cached interpolated values; scales remain stable.
  const beforePlayback = await figure
    .locator(".profile-x .profile-curve")
    .getAttribute("d");
  await page.getByRole("button", { name: "Play", exact: true }).click();
  await expect
    .poll(() => figure.locator(".profile-x .profile-curve").getAttribute("d"))
    .not.toBe(beforePlayback);
  await page.waitForTimeout(180);
  await page.getByRole("button", { name: "Pause", exact: true }).click();
  expect(await figure.locator(".profile-x").getAttribute("data-scale")).toBe(
    scales,
  );
  // D edits refresh the current scientific field; profiles read the returned field.
  const dResponse = page.waitForResponse((r) =>
    r.url().endsWith("/playground/reference"),
  );
  await page.getByLabel("Diffusivity", { exact: true }).fill(".12");
  await dResponse;
  await ready(page);
  const changedD = await at(page, 0.237);
  await profileCheck(figure, changedD);
  expect(changedD).not.toEqual(later);
  const two = await preset(page, "Two hot spots");
  await profileCheck(figure, two.frames[0]);
  await figure.screenshot({ path: path.join(evidence, "two-hot-spots.png") });
  // Hover links coordinates without altering the canvas image or selecting a sensor.
  await figure.locator(".profile-y svg").hover({ position: { x: 10, y: 100 } });
  await expect(figure.locator(".profile-crosshair")).toBeVisible();
  await expect(figure.locator(".profile-readout")).toContainText("y =");
  await page.getByRole("button", { name: "Click points", exact: true }).click();
  await figure.locator("canvas").click({ position: { x: 70, y: 80 } });
  await expect(
    page.getByText("1 sensor points · click/tap to toggle", { exact: true }),
  ).toBeVisible();
  const measurement = page.waitForResponse((r) =>
    r.url().endsWith("/playground/measurements"),
  );
  await page
    .getByRole("button", { name: "Take measurements", exact: true })
    .click();
  expect((await measurement).ok()).toBeTruthy();
  const summary = await page.getByTestId("measurement-summary").innerText(),
    id = await page.locator(".heat-stage").getAttribute("data-reference-id");
  let writes = 0;
  page.on("request", (r) => {
    if (r.method() === "POST") writes++;
  });
  await page.getByLabel("Show profiles", { exact: true }).uncheck();
  await expect(figure.locator(".spatial-profile")).toHaveCount(0);
  await page.getByLabel("Show profiles", { exact: true }).check();
  await page
    .getByLabel("Profile display", { exact: true })
    .selectOption("magnitude");
  await profileCheck(figure, two.frames[0], [1, 1], true);
  await expect(page.getByTestId("measurement-summary")).toHaveText(summary);
  await expect(page.locator(".heat-stage")).toHaveAttribute(
    "data-reference-id",
    id!,
  );
  expect(writes).toBe(0);
  await page
    .getByLabel("Profile display", { exact: true })
    .selectOption("signed");
  await preset(page, "Draw your own");
  const canvas = figure.locator("canvas"),
    b = (await canvas.boundingBox())!;
  const before = await figure
    .locator(".profile-x .profile-curve")
    .getAttribute("d");
  await page.mouse.move(b.x + b.width * 0.3, b.y + b.height * 0.6);
  await page.mouse.down();
  await page.mouse.move(b.x + b.width * 0.45, b.y + b.height * 0.5, {
    steps: 8,
  });
  expect(
    await figure.locator(".profile-x .profile-curve").getAttribute("d"),
  ).not.toBe(before);
  await page.mouse.up();
  await ready(page);
  const drawn = await page.request.get(
    `/api/playground/reference/${await page.locator(".heat-stage").getAttribute("data-reference-id")}/at?time=0`,
  );
  await profileCheck(figure, (await drawn.json()).frame);
  await page.getByLabel("Brush", { exact: true }).selectOption("erase");
  const drawnPath = await figure
    .locator(".profile-x .profile-curve")
    .getAttribute("d");
  await canvas.click({ position: { x: b.width * 0.4, y: b.height * 0.5 } });
  await ready(page);
  expect(
    await figure.locator(".profile-x .profile-curve").getAttribute("d"),
  ).not.toBe(drawnPath);
  const random = await preset(page, "Smooth random");
  await profileCheck(figure, random.frames[0]);
  const randomPath = await figure
    .locator(".profile-x .profile-curve")
    .getAttribute("d");
  const randomResponse = page.waitForResponse((r) =>
    r.url().endsWith("/playground/reference"),
  );
  await page
    .getByRole("button", { name: "New random field", exact: true })
    .click();
  await randomResponse;
  await ready(page);
  expect(
    await figure.locator(".profile-x .profile-curve").getAttribute("d"),
  ).not.toBe(randomPath);
  await preset(page, "Formula");
  async function formula(text: string, lengths = [1, 1]) {
    const r = page.waitForResponse((r) =>
      r.url().endsWith("/playground/reference"),
    );
    await page.getByLabel("Starting field formula", { exact: true }).fill(text);
    const data = await (await r).json();
    await ready(page);
    await profileCheck(figure, data.frames[0], lengths);
    return data;
  }
  await formula("-2");
  const negativeScale = JSON.parse(
    (await figure.locator(".profile-y").getAttribute("data-scale"))!,
  );
  expect(negativeScale[0]).toBeLessThan(-2);
  expect(negativeScale[1]).toBeGreaterThan(0);
  await formula("cos(2*pi*x)*cos(2*pi*y)");
  const cancelPoints = (await figure
    .locator(".profile-x .profile-curve")
    .getAttribute("d"))!
    .match(/[-+\d.e]+,[-+\d.e]+/g)!
    .map((v) => Number(v.split(",")[1]));
  expect(Math.max(...cancelPoints) - Math.min(...cancelPoints)).toBeLessThan(1);
  await page
    .getByLabel("Profile display", { exact: true })
    .selectOption("magnitude");
  await expect(
    figure.getByRole("img", {
      name: "Heat field: Magnitude along x",
      exact: true,
    }),
  ).toBeVisible();
  await page
    .getByLabel("Profile display", { exact: true })
    .selectOption("signed");
  await page.locator(".playground-details>summary").click();
  for (const [label, value] of [
    ["Domain length x", "2"],
    ["Domain length y", "1"],
    ["Domain origin x", "-1"],
    ["Domain origin y", "3"],
  ]) {
    if ((await page.getByLabel(label, { exact: true }).inputValue()) === value)
      continue;
    const r = page.waitForResponse((r) =>
      r.url().endsWith("/playground/reference"),
    );
    await page.getByLabel(label, { exact: true }).fill(value);
    await r;
    await ready(page);
  }
  const nonsquare = await formula("-2", [2, 1]);
  await profileCheck(figure, nonsquare.frames[0], [2, 1]);
  await alignment(figure, 2);
  // Invalid draft domain does not crash the existing rendered field.
  await page.getByLabel("Domain length x", { exact: true }).fill("");
  await expect(figure.locator("canvas")).toBeVisible();
  await page.getByLabel("Domain length x", { exact: true }).fill("2");
  await ready(page);
  await page.setViewportSize({ width: 390, height: 844 });
  await alignment(figure, 2);
  expect(
    await page.evaluate(() => document.documentElement.scrollWidth),
  ).toBeLessThanOrEqual(390);
  await page.screenshot({
    path: path.join(evidence, "mobile-signed.png"),
    fullPage: true,
  });
  expect(errors).toEqual([]);
  fs.writeFileSync(
    path.join(evidence, "browser-fields.json"),
    JSON.stringify({
      spot: spot.frames[0],
      two: two.frames[0],
      later,
      changedD,
      pixelPeak,
      errors,
    }),
  );
});

test("actual checkpoint reconstruction shares reference/model scales without new inference", async ({
  page,
}) => {
  await page.goto("/");
  await ready(page);
  await preset(page, "Hot spot");
  await page
    .getByLabel("Import heat model", { exact: true })
    .setInputFiles(
      path.resolve(
        "../outputs/continuous-time-20261028/local/pipeline/export/coordinate.pt",
      ),
    );
  await expect(page.getByLabel("Heat model", { exact: true })).toBeVisible();
  await page.getByLabel("Barely trained", { exact: false }).check();
  await page
    .getByRole("button", { name: "Take measurements", exact: true })
    .click();
  await expect(page.getByTestId("measurement-summary")).toContainText(
    "measurements",
  );
  let generations = 0;
  page.on("request", (r) => {
    if (r.url().endsWith("/coordinate/reconstruct")) generations++;
  });
  await page.getByRole("button", { name: "Run model", exact: true }).click();
  await expect(page.getByTestId("comparison")).toBeVisible({ timeout: 60000 });
  const ref = page
    .locator(".heat-figure")
    .filter({ has: page.getByRole("img", { name: "Reference", exact: true }) });
  const model = page.locator(".heat-figure").filter({
    has: page.getByRole("img", { name: "Model reconstruction", exact: true }),
  });
  await alignment(ref);
  await alignment(model);
  for (const axis of ["x", "y"]) {
    expect(
      await ref.locator(`.profile-${axis}`).getAttribute("data-scale"),
    ).toBe(await model.locator(`.profile-${axis}`).getAttribute("data-scale"));
  }
  const jobResponse = page.waitForResponse(
    (r) => r.url().includes("/results/") && r.url().endsWith("/at?time=0.237"),
  );
  const currentRef = await at(page, 0.237);
  const decoded = await (await jobResponse).json();
  await profileCheck(ref, currentRef);
  await profileCheck(model, decoded.frame);
  const id = await page
    .locator(".heat-stage")
    .getAttribute("data-reference-id");
  await page
    .getByLabel("Profile display", { exact: true })
    .selectOption("magnitude");
  await profileCheck(model, decoded.frame, [1, 1], true);
  await page.getByLabel("Show profiles", { exact: true }).uncheck();
  await expect(page.getByTestId("comparison")).toBeVisible();
  await page.getByLabel("Show profiles", { exact: true }).check();
  await expect(page.locator(".heat-stage")).toHaveAttribute(
    "data-reference-id",
    id!,
  );
  expect(generations).toBe(1);
  await page.screenshot({
    path: path.join(evidence, "comparison.png"),
    fullPage: true,
  });
});
