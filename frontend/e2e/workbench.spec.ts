import { experts } from "./disclosure";
import { test, expect } from "@playwright/test";

async function setup(page: import("@playwright/test").Page, dimension = 2) {
  await page.goto("/");
  await page.getByRole("button", { name: "Reset draft", exact: true }).click();
  await experts(page);
  await experts(page);
  await page.getByRole("button", { name: /^Heat \/ diffusion/ }).click();
  await experts(page);
  await expect(page.getByLabel("Diffusivity", { exact: true })).toBeVisible();
  if (dimension !== 2) {
    const option = await page
      .getByLabel("Registered checkpoint")
      .locator("option")
      .allTextContents();
    const name = option.find((t) => t.includes(`${dimension}D`))!;
    await page
      .getByLabel("Registered checkpoint")
      .selectOption({ label: name });
    await page.getByRole("button", { name: "Use checkpoint layout" }).click();
    await experts(page);
    await page.getByRole("button", { name: "Apply reviewed changes" }).click();
    await experts(page);
  }
  await page.getByLabel("Diffusivity", { exact: true }).fill("0.08");
  await page
    .getByRole("button", { name: "Validate problem", exact: true })
    .click();
  await experts(page);
  await expect(page.getByText("checkpoint: yes")).toBeVisible();
  await page.screenshot({
    path: "../outputs/workbench-redesign-20261025/browser/regression/browser-symbolic-desktop.png",
    fullPage: true,
  });
  await page.getByRole("button", { name: "Continue to observations" }).click();
  await experts(page);
  await page
    .getByRole("button", { name: "Generate synthetic observations" })
    .click();
  await experts(page);
  await expect(
    page.getByText("40 measurements", { exact: true }),
  ).toBeVisible();
}

test("real 2D reconstruction, export, detached reference, refresh and mobile layout", async ({
  page,
}) => {
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(e.message));
  await setup(page);
  // Export and re-import a real CSV rather than relying only on synthetic rows.
  const downloadPromise = page.waitForEvent("download");
  await page.getByRole("button", { name: "Export CSV" }).click();
  await experts(page);
  const csv = await downloadPromise;
  const path = await csv.path();
  await page.getByLabel("Import observations").setInputFiles(path!);
  await page
    .getByRole("button", { name: "Detach reference", exact: true })
    .click();
  await experts(page);
  await page.getByRole("button", { name: "Configure reconstruction" }).click();
  await experts(page);
  await page.getByLabel("ODE stepper").selectOption("euler");
  await page.getByLabel("Generation schedule").selectOption("quadratic");
  await page.getByLabel("Generation steps", { exact: true }).fill("4");
  await page.getByRole("button", { name: "Validate complete request" }).click();
  await experts(page);
  await expect(page.getByText("Compatible request")).toBeVisible();
  await page
    .getByRole("button", { name: "Run reconstruction", exact: true })
    .click();
  await experts(page);
  await expect(
    page.getByRole("heading", { name: "Inspect the reconstruction" }),
  ).toBeVisible({ timeout: 30000 });
  await expect(
    page.getByText("Physical diagnostics", { exact: true }),
  ).toBeVisible();
  await expect(
    page.getByText("No evaluation reference attached", { exact: true }),
  ).toBeVisible();
  await expect(
    page.getByRole("img", { name: /Reconstruction, x \/ y/ }),
  ).toBeVisible();
  const filePromise = page.waitForEvent("download");
  await page.getByRole("link", { name: "Download numerical results" }).click();
  await experts(page);
  expect((await filePromise).suggestedFilename()).toBe("samples.npz");
  await page
    .getByRole("button", { name: "Export configuration", exact: true })
    .click();
  await experts(page);
  await page.screenshot({
    path: "../outputs/workbench-redesign-20261025/browser/regression/browser-results-desktop.png",
    fullPage: true,
  });
  await page.reload();
  await page.getByRole("button", { name: /^Runs$/ }).click();
  await experts(page);
  await expect(
    page.getByRole("button", { name: "Open run" }).first(),
  ).toBeVisible();
  await page.getByRole("button", { name: "Open run" }).first().click();
  await experts(page);
  await expect(
    page.getByText("Physical diagnostics", { exact: true }),
  ).toBeVisible();
  await page.setViewportSize({ width: 430, height: 900 });
  await page.screenshot({
    path: "../outputs/workbench-redesign-20261025/browser/regression/browser-results-narrow.png",
    fullPage: true,
  });
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= window.innerWidth,
    ),
  ).toBeTruthy();
  expect(errors).toEqual([]);
});

test("true 3D slices and explicitly attached evaluation reference", async ({
  page,
}) => {
  await setup(page, 3);
  await page.getByRole("button", { name: "Configure reconstruction" }).click();
  await experts(page);
  await page.getByLabel("Generation steps", { exact: true }).fill("3");
  await page.getByRole("button", { name: "Validate complete request" }).click();
  await experts(page);
  await page
    .getByRole("button", { name: "Run reconstruction", exact: true })
    .click();
  await experts(page);
  await expect(
    page.getByRole("heading", { name: "Inspect the reconstruction" }),
  ).toBeVisible({ timeout: 30000 });
  await expect(page.getByLabel("Slice normal")).toBeVisible();
  await page.getByLabel("Slice normal").selectOption("0");
  await page.getByLabel("Slice index").fill("3");
  await page.getByLabel("Physical time selection").fill("2");
  await page.getByLabel("Ensemble view").selectOption("-2");
  await expect(
    page.getByRole("img", { name: /Empirical spread, y \/ z/ }).first(),
  ).toBeVisible();
  await page.getByLabel("Ensemble view").selectOption("-1");
  await page
    .getByRole("button", { name: "Attach synthetic reference for evaluation" })
    .click();
  await experts(page);
  await expect(page.getByRole("img", { name: /Absolute error/ })).toBeVisible({
    timeout: 20000,
  });
  await page.screenshot({
    path: "../outputs/workbench-redesign-20261025/browser/regression/browser-3d.png",
    fullPage: true,
  });
});

test("unsupported equation, malformed measurements and cancellation", async ({
  page,
}) => {
  await setup(page);
  await page.getByLabel("Import observations").setInputFiles({
    name: "malformed.csv",
    mimeType: "text/csv",
    buffer: Buffer.from("field,x,value\nu,0,1"),
  });
  await expect(page.getByRole("alert")).toContainText("required columns");
  await page.getByRole("button", { name: "Dismiss" }).click();
  await experts(page);
  await page.getByRole("button", { name: /1\s*Define problem/ }).click();
  await page
    .getByRole("heading", { name: "Define the physical problem", exact: true })
    .waitFor({ state: "visible" });
  await experts(page);
  await page.getByRole("button", { name: /^Poisson/ }).click();
  await experts(page);
  await page.getByRole("button", { name: "Apply reviewed changes" }).click();
  await experts(page);
  await expect(
    page.getByText(/No implemented specialist/).first(),
  ).toBeVisible();
  await page.getByRole("button", { name: "Continue to observations" }).click();
  await experts(page);
  await page.getByRole("button", { name: "Configure reconstruction" }).click();
  await experts(page);
  await page.getByRole("button", { name: "Validate complete request" }).click();
  await experts(page);
  await expect(
    page.getByRole("alert").filter({ hasText: "Action needs attention" }),
  ).toContainText("Current mathematical draft");
  await page.getByRole("button", { name: "Dismiss" }).click();
  await experts(page);
  await page.getByRole("button", { name: /1\s*Define problem/ }).click();
  await page
    .getByRole("heading", { name: "Define the physical problem", exact: true })
    .waitFor({ state: "visible" });
  await experts(page);
  await page.getByRole("button", { name: /^Heat \/ diffusion/ }).click();
  await experts(page);
  await page.getByRole("button", { name: "Apply reviewed changes" }).click();
  await experts(page);
  await expect(page.getByText("checkpoint: yes")).toBeVisible();
  await page.getByRole("button", { name: /3\s*Reconstruct/ }).click();
  if (
    await page
      .getByRole("button", {
        name: "Adjust reconstruction settings",
        exact: true,
      })
      .isVisible()
  )
    await page
      .getByRole("button", {
        name: "Adjust reconstruction settings",
        exact: true,
      })
      .click();
  await experts(page);
  await page.getByLabel("Generation steps", { exact: true }).fill("512");
  await page.getByRole("button", { name: "Validate complete request" }).click();
  await experts(page);
  await page
    .getByRole("button", { name: "Run reconstruction", exact: true })
    .click();
  await experts(page);
  await page.getByRole("button", { name: "Cancel job", exact: true }).click();
  await experts(page);
  await expect(page.getByText("cancelled · cancelled")).toBeVisible();
  await page.getByRole("button", { name: /^Runs$/ }).click();
  await experts(page);
  await expect(
    page.getByRole("cell", { name: "cancelled", exact: true }).first(),
  ).toBeVisible();
});

test("real 1D field curves and sensor overlays", async ({ page }) => {
  await setup(page, 1);
  await page
    .getByRole("button", { name: "Detach reference", exact: true })
    .click();
  await experts(page);
  await page.getByRole("button", { name: "Configure reconstruction" }).click();
  await experts(page);
  await page.getByLabel("Generation steps", { exact: true }).fill("3");
  await page.getByRole("button", { name: "Validate complete request" }).click();
  await experts(page);
  await page
    .getByRole("button", { name: "Run reconstruction", exact: true })
    .click();
  await experts(page);
  await expect(
    page.getByRole("heading", { name: "Inspect the reconstruction" }),
  ).toBeVisible({ timeout: 30000 });
  await expect(
    page.getByRole("img", { name: /Reconstruction, x, physical time/ }),
  ).toBeVisible();
  await expect(page.getByLabel("Slice normal")).toHaveCount(0);
  await page.screenshot({
    path: "../outputs/workbench-redesign-20261025/browser/regression/browser-1d.png",
    fullPage: true,
  });
});

test("real symbolic editing, request roundtrip, draft isolation and keyboard toolbar on narrow layout", async ({
  page,
}) => {
  const external: string[] = [];
  const errors: string[] = [];
  page.on("request", (r) => {
    if (r.url().startsWith("http") && !r.url().includes("127.0.0.1"))
      external.push(r.url());
  });
  page.on("pageerror", (e) => errors.push(e.message));
  await page.goto("/");
  await page.getByRole("button", { name: "Reset draft", exact: true }).click();
  await experts(page);
  await experts(page);
  const field = page.locator("math-field").first();
  await expect(field).toBeVisible();
  expect(
    await field.evaluate((e: HTMLElement & { value: string }) => e.value),
  ).toBe("");
  await page
    .getByRole("button", { name: "Insert fraction in equation 1", exact: true })
    .click();
  expect(
    await field.evaluate((e: HTMLElement & { value: string }) => e.value),
  ).toContain("frac");
  expect(await page.evaluate(() => document.activeElement?.tagName)).toBe(
    "MATH-FIELD",
  );
  // Inserting into the untouched equation collapses preset insertion; reopen it deliberately.
  await experts(page);
  await page.getByRole("button", { name: /^Heat \/ diffusion/ }).click();
  await experts(page);
  await page.getByRole("button", { name: "Apply reviewed changes" }).click();
  await experts(page);
  await expect(page.getByText("checkpoint: yes")).toBeVisible();
  // Actual keyboard editing inside MathLive, without the LaTeX textarea.
  await field.click();
  await experts(page);
  await field.press("End");
  await field.pressSequentially("+u");
  await expect(
    page.getByText(/No implemented specialist/).first(),
  ).toBeVisible();
  const editor = page.getByLabel("Complete mathematical request JSON");
  expect(JSON.parse(await editor.inputValue()).preset).toBeNull();
  await field.press("Backspace");
  await field.press("Backspace");
  await expect(page.getByText("checkpoint: yes")).toBeVisible();
  await page.getByLabel("Diffusivity", { exact: true }).fill(".09");
  await page.getByLabel("Boundary details").fill("Periodic on every face");
  await page.getByLabel("Initial details").fill("Unknown signed initial state");
  await page.getByLabel("Request noise standard deviation").fill(".02");
  await page.getByLabel("x lengths").fill("1");
  await expect(page.getByText("checkpoint: yes")).toBeVisible();
  await page.getByRole("button", { name: "Continue to observations" }).click();
  await experts(page);
  await page
    .getByRole("button", { name: "Generate synthetic observations" })
    .click();
  await experts(page);
  await page.getByRole("button", { name: "Configure reconstruction" }).click();
  await experts(page);
  await page.getByLabel("Generation steps", { exact: true }).fill("2");
  await page.getByRole("button", { name: "Validate complete request" }).click();
  await experts(page);
  await expect(page.getByText("Compatible request")).toBeVisible();
  await page
    .getByRole("button", { name: "Run reconstruction", exact: true })
    .click();
  await experts(page);
  await expect(
    page.getByRole("heading", { name: "Inspect the reconstruction" }),
  ).toBeVisible({ timeout: 30000 });
  await page.getByRole("button", { name: /1\s*Define problem/ }).click();
  await page
    .getByRole("heading", { name: "Define the physical problem", exact: true })
    .waitFor({ state: "visible" });
  await experts(page);
  const before = JSON.parse(await editor.inputValue());
  await page.getByRole("button", { name: /^Poisson/ }).click();
  await experts(page);
  await expect(
    page.getByText("Review mathematical request changes"),
  ).toBeVisible();
  await page.getByRole("button", { name: "Apply reviewed changes" }).click();
  await experts(page);
  await expect(
    page.getByText(/No implemented specialist/).first(),
  ).toBeVisible();
  await page.getByRole("button", { name: "Show LaTeX source" }).click();
  await experts(page);
  await page.getByLabel("LaTeX source 1").fill(String.raw`-\Delta u=f+u`);
  await expect(
    page.getByText("Current source parsed", { exact: false }),
  ).toBeVisible();
  const after = JSON.parse(await editor.inputValue());
  expect(after.measurements).toEqual(before.measurements);
  expect(after.boundary).toEqual(before.boundary);
  expect(after.initial.status).toBe("not_applicable");
  expect(after.domain.times).toEqual([]);
  const downloadEvent = page.waitForEvent("download");
  await page.getByRole("button", { name: "Export complete request" }).click();
  await experts(page);
  const download = await downloadEvent;
  await page
    .getByLabel("Import complete request")
    .setInputFiles((await download.path())!);
  expect(JSON.parse(await editor.inputValue())).toEqual(after);
  await page.getByLabel("LaTeX source 1").fill("u_");
  await expect(
    page.getByText("Current source is not parsed", { exact: false }),
  ).toBeVisible();
  await page.getByRole("button", { name: /3\s*Reconstruct/ }).click();
  if (
    await page
      .getByRole("button", {
        name: "Adjust reconstruction settings",
        exact: true,
      })
      .isVisible()
  )
    await page
      .getByRole("button", {
        name: "Adjust reconstruction settings",
        exact: true,
      })
      .click();
  await experts(page);
  await expect(
    page.getByRole("button", { name: "Run reconstruction", exact: true }),
  ).toBeDisabled();
  await page.getByRole("button", { name: /1\s*Define problem/ }).click();
  await page
    .getByRole("heading", { name: "Define the physical problem", exact: true })
    .waitFor({ state: "visible" });
  await experts(page);
  await page.setViewportSize({ width: 430, height: 900 });
  await expect(field).toBeVisible();
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= window.innerWidth,
    ),
  ).toBe(true);
  await page.screenshot({
    path: "../outputs/workbench-redesign-20261025/browser/regression/browser-symbolic-narrow.png",
    fullPage: true,
  });
  expect(errors).toEqual([]);
  expect(external).toEqual([]);
});
