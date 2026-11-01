import { test, expect } from "@playwright/test";
import type { Page } from "@playwright/test";
const root = "../outputs/coordinate-20261027/browser";
async function inputs(page: Page) {
  await page.goto("/");
  await page.getByRole("button", { name: /^Heat \/ diffusion/ }).click();
  await page.getByRole("button", { name: /2\s*Add information/ }).click();
  await expect(
    page.getByLabel("Input source mode", { exact: true }),
  ).toBeVisible();
}
async function defaults(page: Page) {
  await page
    .getByRole("button", {
      name: "Load visible reference defaults",
      exact: true,
    })
    .click();
}
async function solve(page: Page) {
  const response = page.waitForResponse(
    (r) =>
      r.url().endsWith("/api/references") && r.request().method() === "POST",
  );
  await page
    .getByRole("button", { name: "Solve reference", exact: true })
    .click();
  const r = await response;
  expect(r.status()).toBe(201);
  await expect(
    page.getByRole("img", { name: /Reference field preview/ }),
  ).toBeVisible();
  return (await r.json()).id as string;
}
async function openCoordinate(page: Page) {
  await page
    .getByRole("button", { name: "Reconstruct from measurements", exact: true })
    .click();
  await expect(
    page.getByRole("heading", {
      name: "Resolution-flexible heat reconstruction",
    }),
  ).toBeVisible();
  const option = page
    .getByLabel("Coordinate checkpoint", { exact: true })
    .locator("option")
    .filter({ hasText: "heat-flow" })
    .first();
  await expect(option).toBeAttached();
  await page
    .getByLabel("Coordinate checkpoint", { exact: true })
    .selectOption((await option.getAttribute("value"))!);
  await page.getByLabel("Coordinate flow steps", { exact: true }).fill("2");
}

test("actual analytical field → reproducible noisy observations → latent coordinate inverse → comparison, slices and spread", async ({
  page,
}) => {
  test.setTimeout(90000);
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(e.message));
  await inputs(page);
  await page
    .getByLabel("Input source mode", { exact: true })
    .selectOption("analytical");
  await defaults(page);
  await expect(
    page.getByRole("button", {
      name: "Replace reference inputs with reviewed example",
      exact: true,
    }),
  ).toBeEnabled();
  await page
    .getByRole("button", {
      name: "Replace reference inputs with reviewed example",
      exact: true,
    })
    .click();
  await page
    .getByLabel("Analytical reference expression", { exact: true })
    .fill("1/(x-x)");
  await page
    .getByRole("button", { name: "Solve reference", exact: true })
    .click();
  await expect(
    page
      .getByRole("alert")
      .filter({ hasText: /finite|shape/ })
      .first(),
  ).toBeVisible();
  await page
    .getByRole("button", {
      name: "Replace reference inputs with reviewed example",
      exact: true,
    })
    .click();
  await page
    .getByRole("button", { name: "Verify analytical candidate", exact: true })
    .click();
  await expect(page.getByText(/"passed": true/).first()).toBeVisible();
  await solve(page);
  await page.getByLabel("Sampling count", { exact: true }).fill("3");
  await page
    .getByLabel("Sampling noise mode", { exact: true })
    .selectOption("relative_rms");
  await page.getByLabel("Sampling noise", { exact: true }).fill(".05");
  await expect(
    page.getByText(/Preview total: 12 scalar observations/),
  ).toBeVisible();
  const generated = page.waitForResponse(
    (r) => r.url().endsWith("/sample") && r.request().method() === "POST",
  );
  await page
    .getByRole("button", { name: "Generate observations", exact: true })
    .click();
  const sample = await (await generated).json();
  expect(sample.count).toBe(12);
  expect(sample.actual_sigma).toHaveLength(12);
  expect(sample.rows[0]).toHaveProperty("sigma");
  await expect(page.getByText(/12 noisy measurements/)).toBeVisible();
  await page
    .getByText("Observed values and absolute noise", { exact: true })
    .click();
  await expect(
    page.getByRole("table", { name: "Generated observation values" }),
  ).toBeVisible();
  await page.screenshot({ path: root + "/observations.png", fullPage: true });
  await openCoordinate(page);
  await expect(
    page.getByLabel("Coordinate compute device", { exact: true }),
  ).toHaveValue("cpu");
  await page.getByLabel("Coordinate query grid", { exact: true }).fill("12,12");
  let sent: Record<string, unknown> | null = null;
  page.on("request", (r) => {
    if (r.url().endsWith("/coordinate/reconstruct") && r.method() === "POST")
      sent = r.postDataJSON();
  });
  await page
    .getByRole("button", {
      name: "Reconstruct with coordinate specialist",
      exact: true,
    })
    .click();
  await expect(
    page.getByRole("status").filter({ hasText: /Coordinate job.*succeeded/ }),
  ).toBeVisible({ timeout: 45000 });
  expect(sent).toBeTruthy();
  expect(sent).toHaveProperty("device", "cpu");
  expect(sent).not.toHaveProperty("reference_id");
  expect(sent).not.toHaveProperty("inputs");
  expect(sent).not.toHaveProperty("analytical_reference");
  await page
    .getByRole("button", {
      name: "Compare coordinate result with reference",
      exact: true,
    })
    .click();
  await expect(page.getByRole("img", { name: /^Reference,/ })).toBeVisible();
  await expect(
    page.getByRole("img", { name: /^Coordinate reconstruction,/ }),
  ).toBeVisible();
  await page.getByLabel("Coordinate physical time", { exact: true }).fill("2");
  await page.screenshot({ path: root + "/comparison.png", fullPage: true });
  await page
    .getByLabel("Coordinate statistic", { exact: true })
    .selectOption("spread");
  await page
    .getByLabel("Coordinate statistic", { exact: true })
    .selectOption("sample:0");
  await page.reload();
  await expect(
    page.getByRole("status").filter({ hasText: /Coordinate job.*succeeded/ }),
  ).toBeVisible();
  await expect(
    page.getByRole("img", { name: /^Coordinate reconstruction,/ }),
  ).toBeVisible();
  const text = await page
    .getByRole("status")
    .filter({ hasText: /Coordinate job.*succeeded/ })
    .textContent();
  const jobId = text!.match(/Coordinate job ([a-f0-9]{32})/)![1];
  await page.getByRole("button", { name: "Runs", exact: true }).click();
  await page.getByRole("button", { name: "Refresh", exact: true }).click();
  const row = page.getByRole("row").filter({ hasText: jobId.slice(0, 10) });
  await expect(row).toContainText("Coordinate heat");
  await row.getByRole("button", { name: "Open run", exact: true }).click();
  await expect(
    page.getByRole("status").filter({ hasText: new RegExp(jobId) }),
  ).toBeVisible();
  await expect(
    page.getByRole("img", { name: /^Coordinate reconstruction,/ }),
  ).toBeVisible();
  await page
    .getByRole("button", { name: "Models & Training", exact: true })
    .click();
  await page
    .getByText("Latent coordinate heat specialist · two training stages", {
      exact: true,
    })
    .click();
  await expect(
    page.getByRole("table", {
      name: "Coordinate training stages",
      exact: true,
    }),
  ).toContainText("heat-flow");
  await expect
    .poll(() => page.evaluate(() => document.documentElement.scrollWidth))
    .toBeLessThanOrEqual(1440);
  await page.screenshot({
    path: root + "/training-stages.png",
    fullPage: true,
  });
  expect(errors).toEqual([]);
});

test("actual numerical full-field import, count rejection, measurements-only zeros and scattered sensor reconstruction", async ({
  page,
  request,
}) => {
  test.setTimeout(90000);
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(e.message));
  await inputs(page);
  await defaults(page);
  const reference = await solve(page);
  const artifact = await (
    await request.get(
      `http://127.0.0.1:8000/api/references/${reference}/artifacts/reference-field.json`,
    )
  ).json();
  await page
    .getByText("Import complete numerical reference JSON", { exact: true })
    .click();
  await page.getByLabel("Numerical reference artifact", { exact: true }).fill(
    JSON.stringify({
      ...artifact,
      coordinates: { ...artifact.coordinates, x: [0, 0] },
    }),
  );
  await page
    .getByRole("button", {
      name: "Import complete reference field",
      exact: true,
    })
    .click();
  await expect(
    page
      .getByRole("alert")
      .filter({ hasText: /unique|coordinates/ })
      .first(),
  ).toBeVisible();
  await page
    .getByLabel("Numerical reference artifact", { exact: true })
    .fill(JSON.stringify(artifact));
  await page
    .getByRole("button", {
      name: "Import complete reference field",
      exact: true,
    })
    .click();
  await page.getByLabel("Sampling count", { exact: true }).fill("1000");
  await page
    .getByRole("button", { name: "Generate observations", exact: true })
    .click();
  await expect(
    page
      .getByRole("alert")
      .filter({ hasText: /count exceeds|Spatial count/i })
      .first(),
  ).toBeVisible();
  await page
    .getByLabel("Input source mode", { exact: true })
    .selectOption("measurements");
  await page
    .getByLabel("Measurements-only rows", { exact: true })
    .fill('[{"field":"u","x":0,"y":0,"time":0,"value":0,"sigma":0.02}]');
  await page
    .getByRole("button", {
      name: "Adopt measurements without reference",
      exact: true,
    })
    .click();
  await page
    .getByRole("button", { name: "Open reconstruction", exact: true })
    .click();
  const option = page
    .getByLabel("Coordinate checkpoint", { exact: true })
    .locator("option")
    .filter({ hasText: "heat-flow" })
    .first();
  await expect(option).toBeAttached();
  await page
    .getByLabel("Coordinate checkpoint", { exact: true })
    .selectOption((await option.getAttribute("value"))!);
  await page.getByLabel("Coordinate flow steps", { exact: true }).fill("2");
  await page
    .getByText("Scattered measurements or zero-observation prior", {
      exact: true,
    })
    .click();
  await page
    .getByLabel("Coordinate observation rows", { exact: true })
    .fill('[{"field":"u","x":2,"y":0.234,"time":0.03,"value":0,"sigma":0.02}]');
  await page
    .getByRole("button", {
      name: "Reconstruct with coordinate specialist",
      exact: true,
    })
    .click();
  await expect(
    page
      .getByRole("alert")
      .filter({ hasText: /outside physical domain/ })
      .first(),
  ).toBeVisible();
  await page
    .getByLabel("Coordinate observation rows", { exact: true })
    .fill(
      '[{"field":"u","x":0.123,"y":0.234,"time":0.03,"value":0,"sigma":0.02}]',
    );
  await page
    .getByRole("button", {
      name: "Reconstruct with coordinate specialist",
      exact: true,
    })
    .click();
  await expect(
    page.getByRole("status").filter({ hasText: /Coordinate job.*succeeded/ }),
  ).toBeVisible({ timeout: 45000 });
  await expect(
    page.getByRole("button", {
      name: "Compare coordinate result with reference",
      exact: true,
    }),
  ).toBeDisabled();
  await page.screenshot({
    path: root + "/measurements-only.png",
    fullPage: true,
  });
  expect(errors).toEqual([]);
});

test("retained fixed-grid heat workflow still performs numerical inference and comparison", async ({
  page,
}) => {
  test.setTimeout(90000);
  await page.goto("/");
  await page.getByRole("button", { name: /^Heat \/ diffusion/ }).click();
  await page
    .getByText("Edit declarations, domain, conditions & model override", {
      exact: true,
    })
    .click();
  await page.getByLabel("x nodes", { exact: true }).fill("32");
  await page.getByLabel("y nodes", { exact: true }).fill("32");
  await expect(
    page.getByText("checkpoint: yes", { exact: true }),
  ).toBeVisible();
  await page.getByRole("button", { name: /2\s*Add information/ }).click();
  await defaults(page);
  await solve(page);
  await page.getByLabel("Sampling count", { exact: true }).fill("2");
  await page
    .getByRole("button", { name: "Generate observations", exact: true })
    .click();
  await page
    .getByRole("button", { name: "Reconstruct from measurements", exact: true })
    .click();
  await page.getByText("Numerical sampling settings", { exact: true }).click();
  await page.getByLabel("Generation steps", { exact: true }).fill("2");
  await page.getByLabel("Sample count", { exact: true }).fill("2");
  await page
    .getByRole("button", { name: "Validate complete request", exact: true })
    .click();
  await expect(page.getByText(/^Compatible request/)).toBeVisible();
  await page
    .getByRole("button", { name: "Run reconstruction", exact: true })
    .click();
  await expect(
    page.getByRole("heading", {
      name: "Inspect the reconstruction",
      exact: true,
    }),
  ).toBeVisible({ timeout: 60000 });
  await page.getByRole("button", { name: "Compare", exact: true }).click();
  await expect(
    page.getByRole("table", {
      name: "Reference comparison errors",
      exact: true,
    }),
  ).toBeVisible();
  await page.screenshot({ path: root + "/fixed-baseline.png", fullPage: true });
});
