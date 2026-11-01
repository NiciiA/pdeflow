import { test, expect } from "@playwright/test";
import fs from "node:fs";
import path from "node:path";
const out = path.resolve("../outputs/measurement-times-20261101");
test("rendered explicit counts, timestamp replay and manual sensor times", async ({
  page,
}) => {
  await page.goto("/");
  const take = page.getByRole("button", {
    name: "Take measurements",
    exact: true,
  });
  await expect(take).toBeEnabled();
  await page.getByLabel("Measurement-time policy").selectOption("chosen");
  await page
    .getByLabel("Chosen physical times", { exact: true })
    .fill(".02,.2,.47,.83");
  await page.getByLabel("Counts per measurement time").fill("1,2,3,1");
  await expect(page.getByText(/Resolved total:/)).toContainText(
    "7 measurements",
  );
  const response = page.waitForResponse((r) =>
    r.url().endsWith("/playground/measurements"),
  );
  await take.click();
  const record = await (await response).json();
  expect(record.rows).toHaveLength(7);
  await page
    .getByText("Inspect sensor timestamps and values", { exact: true })
    .click();
  await expect(page.getByTestId("sensor-time-0")).toHaveText(
    String(record.rows[0].time),
  );
  await page.locator(".playground-details>summary").click();
  await page.getByLabel("Import measurements", { exact: true }).setInputFiles({
    name: "replay.json",
    mimeType: "application/json",
    buffer: Buffer.from(JSON.stringify(record)),
  });
  await expect(page.getByTestId("sensor-time-0")).toHaveText(
    String(record.rows[0].time),
  );
  await page.screenshot({
    path: path.join(out, "browser-time-inspector.png"),
    fullPage: true,
  });
  await page.reload();
  await expect(take).toBeEnabled();
  await page.getByRole("button", { name: "Click points", exact: true }).click();
  await expect(page.getByLabel("Active measurement time")).toHaveValue("0");
  await page.getByLabel("Active measurement time").fill(".237");
  const canvas = page.getByRole("img", { name: "heat u", exact: true });
  const box = (await canvas.boundingBox())!;
  await page.mouse.click(box.x + box.width * 0.3, box.y + box.height * 0.4);
  await page.getByLabel("Physical time", { exact: true }).evaluate((node) => {
    const e = node as HTMLInputElement;
    Object.getOwnPropertyDescriptor(
      HTMLInputElement.prototype,
      "value",
    )!.set!.call(e, "0.8");
    e.dispatchEvent(new Event("input", { bubbles: true }));
  });
  await expect(page.getByLabel("Active measurement time")).toHaveValue(
    /0?\.237/,
  );
  const manual = page.waitForResponse((r) =>
    r.url().endsWith("/playground/measurements"),
  );
  await take.click();
  const result = await (await manual).json();
  expect(result.rows).toHaveLength(1);
  expect(result.rows[0].time).toBe(0.237);
  fs.writeFileSync(
    path.join(out, "browser-controls-evidence.json"),
    JSON.stringify({ explicit: record, manual: result }, null, 2),
  );
  await page.screenshot({
    path: path.join(out, "browser-manual-time.png"),
    fullPage: true,
  });
});

test("wave velocity timestamp import and missing-time rejection", async ({
  page,
}) => {
  await page.goto("/");
  const take = page.getByRole("button", {
    name: "Take measurements",
    exact: true,
  });
  await expect(take).toBeEnabled();
  await page.getByLabel("Equation", { exact: true }).selectOption("wave");
  await expect(take).toBeEnabled();
  await page
    .getByLabel("Measurement quantity", { exact: true })
    .selectOption("v");
  await page.getByLabel("Number of measurements", { exact: true }).fill("4");
  await page.getByLabel("Measurement-time policy").selectOption("chosen");
  await page
    .getByLabel("Chosen physical times", { exact: true })
    .fill(".02,.2,.47,.83");
  const response = page.waitForResponse((r) =>
    r.url().endsWith("/playground/measurements"),
  );
  await take.click();
  const record = await (await response).json();
  expect(
    record.rows.every((r: { field: string }) => r.field === "v"),
  ).toBeTruthy();
  await page.locator(".playground-details>summary").click();
  const input = page.getByLabel("Import measurements", { exact: true });
  const invalid = JSON.parse(JSON.stringify(record));
  delete invalid.rows[0].time;
  await input.setInputFiles({
    name: "invalid.json",
    mimeType: "application/json",
    buffer: Buffer.from(JSON.stringify(invalid)),
  });
  await expect(page.getByRole("alert")).toContainText("finite x, y, time");
  await input.setInputFiles({
    name: "wave-replay.json",
    mimeType: "application/json",
    buffer: Buffer.from(JSON.stringify(record)),
  });
  await page
    .getByText("Inspect sensor timestamps and values", { exact: true })
    .click();
  await expect(page.getByTestId("sensor-time-0")).toHaveText(
    String(record.rows[0].time),
  );
  await page
    .getByLabel("Observed measurement time", { exact: true })
    .fill(".333");
  await page.getByLabel("Observed value", { exact: true }).fill("3.14");
  await page
    .getByRole("button", { name: "Add observed value", exact: true })
    .click();
  await expect(page.getByTestId("sensor-time-4")).toHaveText("0.333");
  await page.screenshot({
    path: path.join(out, "browser-wave-import-times.png"),
    fullPage: true,
  });
  fs.writeFileSync(
    path.join(out, "browser-wave-import-evidence.json"),
    JSON.stringify(record, null, 2),
  );
});
