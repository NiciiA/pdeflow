import { test, expect } from "@playwright/test";
import fs from "node:fs";
import path from "node:path";
const out = path.resolve("../outputs/training-wiring-20261030");
test("real focused training page routes independent equation choices and field loss", async ({
  page,
}) => {
  const errors: string[] = [];
  let starts = 0;
  page.on("pageerror", (e) => errors.push(e.message));
  page.on("request", (r) => {
    if (r.url().endsWith("/training/start")) starts++;
  });
  await page.goto("/");
  await expect(page.getByLabel("Equation", { exact: true })).toHaveValue(
    "heat",
  );
  await page
    .getByRole("navigation", { name: "Application" })
    .getByRole("button", { name: "Train model", exact: true })
    .click();
  const selector = page.getByLabel("Training equations", { exact: true });
  await expect(selector).toHaveValue("heat-wave-2d");
  const summary = page.getByTestId("resolved-training-summary");
  await expect(summary).toContainText("Heat 50% · Wave 50%");
  await expect(
    page.getByLabel("Field reconstruction loss", { exact: true }),
  ).toHaveValue("mse");
  await page
    .getByLabel("Field reconstruction loss", { exact: true })
    .selectOption("region_balanced_mse");
  await expect(
    page.getByLabel("Feature balance: α", { exact: true }),
  ).toHaveValue("0.5");
  const evidence: unknown[] = [];
  for (const [choice, families] of [
    ["heat-2d", ["heat"]],
    ["wave-2d", ["wave"]],
    ["heat-wave-2d", ["heat", "wave"]],
  ] as const) {
    const response = page.waitForResponse(
      (r) =>
        r.url().endsWith("/training/plan") &&
        r.request().postDataJSON()?.problem === choice,
    );
    await selector.selectOption(choice);
    const r = await response;
    expect(r.ok()).toBeTruthy();
    const resolved = await r.json();
    const payload = r.request().postDataJSON();
    expect(Object.keys(resolved.config.family_weights).sort()).toEqual([
      ...families,
    ]);
    expect(Object.keys(resolved.training_physics.state_schemas).sort()).toEqual(
      [...families],
    );
    expect(payload.advanced.field_loss).toMatchObject({
      type: "region_balanced_mse",
      alpha: 0.5,
    });
    expect(resolved.config.field_loss).toMatchObject({
      type: "region_balanced_mse",
      alpha: 0.5,
    });
    await expect(summary).toHaveAttribute("aria-busy", "false");
    if (choice === "heat-2d") {
      await expect(summary).toContainText("Heat 100%");
      await expect(summary).not.toContainText("Wave state");
      await expect(summary).not.toContainText("wave speed");
    }
    if (choice === "wave-2d") {
      await expect(summary).toContainText("Wave 100%");
      await expect(summary).not.toContainText("Heat state");
      await expect(summary).not.toContainText("diffusivity");
    }
    if (choice === "heat-wave-2d") {
      await expect(summary).toContainText("∂u/∂t = D Δu");
      await expect(summary).toContainText("∂²u/∂t² = c² Δu");
      await expect(summary).toContainText("Heat state: u.");
      await expect(summary).toContainText(
        "Wave state: displacement u and velocity v",
      );
    }
    await expect(summary).toContainText(
      "Initial-field mixture types are not PDE families",
    );
    evidence.push({ choice, payload, resolved });
  }
  await page.getByLabel("Feature balance: α", { exact: true }).fill("0.65");
  const changed = page.waitForResponse(
    (r) =>
      r.url().endsWith("/training/plan") &&
      r.request().postDataJSON()?.problem === "wave-2d" &&
      r.request().postDataJSON()?.advanced?.field_loss?.alpha === 0.65,
  );
  await selector.selectOption("wave-2d");
  expect((await changed).ok()).toBeTruthy();
  await expect(
    page.getByLabel("Feature balance: α", { exact: true }),
  ).toHaveValue("0.65");
  await page.getByLabel("Feature balance: α", { exact: true }).fill("0.5");
  await selector.selectOption("heat-wave-2d");
  await expect(summary).toContainText("Heat 50% · Wave 50%");
  await expect(summary).toHaveAttribute("aria-busy", "false");
  await expect(page.getByText(/A2 uses the same decoded/)).toBeVisible();
  await expect(
    page.getByText("Loading saved training runs…", { exact: true }),
  ).not.toBeVisible({ timeout: 30000 });
  await page
    .locator(".integrated-training form")
    .getByRole("button", { name: "Train model", exact: true })
    .waitFor();
  await page
    .locator(".integrated-training form")
    .screenshot({ path: path.join(out, "joint-training-summary-detail.png") });
  await page.screenshot({
    path: path.join(out, "joint-training-summary.png"),
    fullPage: true,
  });
  expect(starts).toBe(0);
  expect(errors).toEqual([]);
  fs.writeFileSync(
    path.join(out, "browser-training-routing.json"),
    JSON.stringify({ evidence, starts, errors }, null, 2),
  );
});
