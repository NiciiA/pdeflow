import { test, expect } from "@playwright/test";
import type { Page } from "@playwright/test";
const root = "../outputs/reference-suite-20261026/browser";
async function expand(page: Page, name: string) {
  const d = page
    .locator("details")
    .filter({ has: page.locator("summary").getByText(name, { exact: true }) });
  if (!(await d.getAttribute("open")))
    await page.getByText(name, { exact: true }).click();
}
async function define(page: Page, family: string, dimension: number) {
  await page.goto("/");
  await page
    .getByRole("button", {
      name:
        family === "heat"
          ? /^Heat \/ diffusion/
          : family === "darcy"
            ? /^Darcy/
            : /^A\+B/,
    })
    .click();
  if (dimension !== 2) {
    await expand(
      page,
      "Edit declarations, domain, conditions & model override",
    );
    await page
      .getByLabel("Spatial dimension", { exact: true })
      .selectOption(String(dimension));
  }
  if (dimension === 1)
    await page.getByLabel("x nodes", { exact: true }).fill("16");
  if (family === "darcy")
    await page
      .getByRole("button", {
        name: "Set known mean-zero sinusoidal forcing",
        exact: true,
      })
      .click();
  await expect(
    page.getByText("checkpoint: yes", { exact: true }),
  ).toBeVisible();
  await page.getByRole("button", { name: /2\s*Add information/ }).click();
  await page
    .getByRole("button", {
      name: "Load visible reference defaults",
      exact: true,
    })
    .click();
  await page
    .getByRole("button", { name: "Preview supplied field", exact: true })
    .click();
  await expect(
    page.getByRole("img", { name: /Supplied reference input/ }),
  ).toBeVisible();
  await page
    .getByRole("button", { name: "Solve reference", exact: true })
    .click();
  await expect(page.getByText(/Immutable reference/)).toBeVisible();
  await expect(
    page.getByRole("img", { name: /Reference field preview/ }),
  ).toBeVisible();
}
for (const [family, dimension] of [
  ["heat", 1],
  ["heat", 2],
  ["darcy", 2],
  ["darcy", 3],
  ["chemistry", 2],
  ["chemistry", 3],
] as const) {
  test(`real ${family} ${dimension}D reference → measurements → reconstruction → compare`, async ({
    page,
  }) => {
    test.setTimeout(120000);
    const errors: string[] = [];
    page.on("pageerror", (e) => errors.push(e.message));
    await define(page, family, dimension);
    if (family === "darcy") {
      await page
        .getByLabel("Disclosure a", { exact: true })
        .selectOption("unknown");
      await page.getByRole("checkbox", { name: "a", exact: true }).uncheck();
      await expect(
        page.getByLabel("Reference physical time", { exact: true }),
      ).toHaveCount(0);
    }
    if (family === "chemistry") {
      for (const n of ["b", "c"]) {
        await page
          .getByLabel(`Disclosure ${n}`, { exact: true })
          .selectOption("unknown");
        await page.getByRole("checkbox", { name: n, exact: true }).uncheck();
      }
    }
    await page.getByLabel("Sampling count", { exact: true }).fill("3");
    await page
      .getByRole("button", { name: "Generate observations", exact: true })
      .click();
    await expect(page.getByText(/noisy measurements\. Clean/)).toBeVisible();
    await expect(
      page.getByRole("table", { name: "Measurement coverage", exact: true }),
    ).toBeVisible();
    if (dimension === 3) {
      await page
        .getByLabel("Reference slice normal", { exact: true })
        .selectOption("0");
      await page.getByLabel("Reference slice index", { exact: true }).fill("3");
      await page
        .getByLabel("Reference field", { exact: true })
        .selectOption("1");
    }
    await page
      .getByRole("checkbox", { name: /Show reference truth/ })
      .uncheck();
    await expect(page.getByText(/Truth hidden/)).toBeVisible();
    await page.screenshot({
      path: `${root}/${family}-${dimension}d-observations.png`,
      fullPage: true,
    });
    await page
      .getByRole("button", {
        name: "Reconstruct from measurements",
        exact: true,
      })
      .click();
    await expand(page, "Numerical sampling settings");
    await page.getByLabel("Generation steps", { exact: true }).fill("2");
    await page.getByLabel("Sample count", { exact: true }).fill("2");
    await page
      .getByRole("button", { name: "Validate complete request", exact: true })
      .click();
    await expect(page.getByText("Compatible request")).toBeVisible();
    let workerBody: Record<string, unknown> | null = null;
    page.on("request", (r) => {
      if (r.url().endsWith("/api/jobs") && r.method() === "POST")
        workerBody = r.postDataJSON();
    });
    await page
      .getByRole("button", { name: "Run reconstruction", exact: true })
      .click();
    await expect(
      page.getByRole("heading", {
        name: "Inspect the reconstruction",
        exact: true,
      }),
    ).toBeVisible({ timeout: 90000 });
    expect(workerBody).toBeTruthy();
    expect(workerBody).not.toHaveProperty("reference_id");
    expect(workerBody).not.toHaveProperty("inputs");
    await page.getByRole("button", { name: "Compare", exact: true }).click();
    await expect(
      page.getByRole("table", {
        name: "Reference comparison errors",
        exact: true,
      }),
    ).toBeVisible({ timeout: 30000 });
    await expect(page.getByRole("img", { name: /Reference,/ })).toBeVisible();
    if (family === "darcy") {
      await expect(
        page.getByRole("table", {
          name: "Reference comparison errors",
          exact: true,
        }),
      ).toContainText("log_a");
      await expect(
        page.getByLabel("Physical time τ", { exact: true }),
      ).toHaveCount(0);
    }
    await page.screenshot({
      path: `${root}/${family}-${dimension}d-comparison.png`,
      fullPage: true,
    });
    if (family === "heat" && dimension === 2) {
      await page.setViewportSize({ width: 430, height: 900 });
      await page.screenshot({
        path: `${root}/heat-narrow.png`,
        fullPage: true,
      });
      expect(
        await page.evaluate(() => document.documentElement.scrollWidth),
      ).toBeLessThanOrEqual(431);
    }
    expect(errors).toEqual([]);
  });
}
test("changed reference blocks sampling; unsupported exact solver and disclosure remain actionable", async ({
  page,
}) => {
  await define(page, "darcy", 2);
  await page
    .getByLabel("Reference solver", { exact: true })
    .selectOption("analytical");
  await page
    .getByRole("button", { name: "Solve reference", exact: true })
    .click();
  await expect(
    page.getByRole("alert").filter({ hasText: "No analytical" }),
  ).toBeVisible();
  await expect(
    page.getByRole("button", { name: "Generate observations", exact: true }),
  ).toBeDisabled();
  await page
    .getByLabel("Reference solver", { exact: true })
    .selectOption("numerical");
  await page
    .getByRole("button", { name: "Solve reference", exact: true })
    .click();
  await expect(
    page.getByRole("button", { name: "Generate observations", exact: true }),
  ).toBeEnabled();
  await page
    .getByLabel("Disclosure a", { exact: true })
    .selectOption("fully_known");
  await page
    .getByRole("button", { name: "Generate observations", exact: true })
    .click();
  await expect(
    page.getByRole("alert").filter({ hasText: "cannot treat full a" }),
  ).toBeVisible();
});

test("managed suite checkpoints and child history are discovered without launching training", async ({
  page,
}) => {
  await page.goto("/");
  await page
    .getByRole("button", { name: "Models & Training", exact: true })
    .click();
  await expect(
    page.getByRole("heading", { name: "Models & Training", exact: true }),
  ).toBeVisible();
  await expect(
    page.getByText("Sequential training suites", { exact: true }),
  ).toBeVisible();
  await page
    .locator("summary")
    .filter({ hasText: /quickstart.*completed/ })
    .first()
    .click();
  await expect(page.getByRole("table", { name: /Suite / })).toContainText(
    "darcy-3d",
  );
  await page
    .getByRole("table", { name: /Suite / })
    .getByRole("button")
    .first()
    .click();
  await expect(
    page.getByRole("img", { name: /Training and validation loss/ }),
  ).toBeVisible();
  await page.screenshot({
    path: root + "/suite-discovery.png",
    fullPage: true,
  });
});

test("reference-free manual CSV observations still reconstruct", async ({
  page,
}) => {
  await page.goto("/");
  await page.getByRole("button", { name: /^Heat \/ diffusion/ }).click();
  await expect(
    page.getByText("checkpoint: yes", { exact: true }),
  ).toBeVisible();
  await page.getByRole("button", { name: /2\s*Add information/ }).click();
  const parsed = page.waitForResponse(
    (r) =>
      r.url().endsWith("/api/mathematics/parse") &&
      r.request().method() === "POST",
  );
  await page
    .getByLabel("Import observations", { exact: true })
    .setInputFiles({
      name: "manual.csv",
      mimeType: "text/csv",
      buffer: Buffer.from(
        "field,x,y,time,value\nu,0,0,0,0.8\nu,0.25,0.5,0.05,0.3\nu,0.5,0.25,0.1,0.1\n",
      ),
    });
  await parsed;
  await page.waitForTimeout(150);
  await page
    .getByRole("button", { name: "Configure reconstruction", exact: true })
    .click();
  await expand(page, "Numerical sampling settings");
  await page.getByLabel("Generation steps", { exact: true }).fill("2");
  await page.getByLabel("Sample count", { exact: true }).fill("1");
  await page
    .getByRole("button", { name: "Validate complete request", exact: true })
    .click();
  await expect(page.getByText("Compatible request")).toBeVisible();
  await page
    .getByRole("button", { name: "Run reconstruction", exact: true })
    .click();
  await expect(
    page.getByRole("heading", {
      name: "Inspect the reconstruction",
      exact: true,
    }),
  ).toBeVisible({ timeout: 60000 });
  await expect(
    page.getByRole("button", { name: "Compare", exact: true }),
  ).toHaveCount(0);
  await expect(
    page.getByText("No evaluation reference attached", { exact: true }),
  ).toBeVisible();
});
