import { test, expect } from "@playwright/test";
import type { Page } from "@playwright/test";
const root = "../outputs/workbench-redesign-20261025/browser";
async function expand(page: Page, name: string) {
  const d = page
    .locator("details")
    .filter({ has: page.locator("summary").getByText(name, { exact: true }) });
  if (!(await d.getAttribute("open")))
    await page.getByText(name, { exact: true }).click();
}
async function define(page: Page, family = "heat", dimension = 2) {
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
}
async function information(page: Page) {
  await page.getByRole("button", { name: /2\s*Add information/ }).click();
  await page
    .getByRole("button", { name: "Generate synthetic observations" })
    .click();
  await expect(page.getByRole("img", { name: /Input preview/ })).toBeVisible();
}
async function configure(page: Page, steps = "3") {
  await page.getByRole("button", { name: "Configure reconstruction" }).click();
  await expand(page, "Numerical sampling settings");
  await page.getByLabel("Generation steps", { exact: true }).fill(steps);
  await page.getByLabel("Sample count", { exact: true }).fill("2");
}
async function run(page: Page) {
  await page.getByRole("button", { name: "Validate complete request" }).click();
  await expect(page.getByText("Compatible request")).toBeVisible();
  await page
    .getByRole("button", { name: "Run reconstruction", exact: true })
    .click();
  await expect(
    page.getByRole("heading", { name: "Inspect the reconstruction" }),
  ).toBeVisible({ timeout: 60000 });
  await expect(
    page.getByRole("img", { name: /Reconstruction,/ }),
  ).toBeVisible();
  await expect(
    page
      .getByRole("img", {
        name: /predicted versus observed with identity line/,
      })
      .first(),
  ).toBeVisible();
}
test("A new user: real guided example, visible inputs, reconstruction and responsive scientific panels", async ({
  page,
}) => {
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(e.message));
  await page.goto("/");
  await page
    .getByRole("button", { name: "Guided example", exact: true })
    .click();
  await page
    .getByRole("button", { name: "Load supported heat example" })
    .click();
  await page.getByRole("button", { name: "Dismiss example" }).click();
  await expect(
    page.getByText(
      "The field spreads over time. D controls the diffusion rate.",
    ),
  ).toBeVisible();
  await expect(page.getByLabel("Registered checkpoint")).not.toBeVisible();
  await expect(
    page.getByRole("region", { name: "Full problem summary" }),
  ).toContainText("Review defaults");
  await expect(
    page.getByRole("img", { name: /unknown regions with measured points/ }),
  ).toBeVisible();
  await page.screenshot({ path: root + "/define-desktop.png", fullPage: true });
  await information(page);
  await page
    .getByRole("button", { name: "Select observation row 1", exact: true })
    .click();
  await expect(
    page.getByRole("button", { name: /Select observation 1:/ }),
  ).toBeVisible();
  await page.screenshot({
    path: root + "/observations-desktop.png",
    fullPage: true,
  });
  await configure(page);
  await run(page);
  await expect(
    page.getByText("No evaluation reference attached", { exact: true }),
  ).toBeVisible();
  await expect(page.getByText(/Finished execution is separate/)).toBeVisible();
  await page
    .getByRole("heading", { name: "Reconstruction", exact: true })
    .scrollIntoViewIfNeeded();
  await page.screenshot({ path: root + "/results-focus.png" });
  await page.screenshot({
    path: root + "/results-desktop.png",
    fullPage: true,
  });
  await page.setViewportSize({ width: 430, height: 900 });
  await expect
    .poll(() =>
      page.evaluate(
        () => document.documentElement.scrollWidth <= window.innerWidth,
      ),
    )
    .toBeTruthy();
  await page
    .getByRole("heading", { name: "Reconstruction", exact: true })
    .scrollIntoViewIfNeeded();
  await page.screenshot({ path: root + "/results-narrow-focus.png" });
  await page.screenshot({ path: root + "/results-narrow.png", fullPage: true });
  expect(errors).toEqual([]);
});
test("B custom equation rearrangement, parameter edit and real CSV import remain executable", async ({
  page,
}) => {
  await define(page);
  await page.getByRole("button", { name: "Show LaTeX source" }).click();
  await page.getByLabel("LaTeX source 1").fill("u_τ-D lap(u)=0");
  await page.getByLabel("Diffusivity", { exact: true }).fill(".09");
  await expect(
    page.getByText("checkpoint: yes", { exact: true }),
  ).toBeVisible();
  await information(page);
  const download = page.waitForEvent("download");
  await page.getByRole("button", { name: "Export CSV" }).click();
  await page
    .getByLabel("Import observations")
    .setInputFiles((await (await download).path())!);
  await page
    .getByRole("button", { name: "Detach reference", exact: true })
    .click();
  await configure(page);
  await run(page);
});
test("C unsupported and invalid source persist across refresh and block stale execution", async ({
  page,
}) => {
  await define(page);
  await page.getByRole("button", { name: "Show LaTeX source" }).click();
  await page.getByLabel("LaTeX source 1").fill("u_τ=D lap(u)+u");
  await expect(
    page.getByText(/No implemented specialist/).first(),
  ).toBeVisible();
  await page.reload();
  await page.getByRole("button", { name: "Show LaTeX source" }).click();
  await expect(page.getByLabel("LaTeX source 1")).toHaveValue("u_τ=D lap(u)+u");
  await page.getByRole("button", { name: /3\s*Reconstruct/ }).click();
  await expect(
    page.getByRole("button", { name: "Run reconstruction", exact: true }),
  ).toBeDisabled();
  await page.getByRole("button", { name: /1\s*Define/ }).click();
  await expand(page, "Request JSON, import & export");
  await page.getByLabel("Complete mathematical request JSON").fill("{bad");
  await page.reload();
  await expect(
    page.getByLabel("Complete mathematical request JSON"),
  ).toHaveValue("{bad");
  await expand(page, "Edit declarations, domain, conditions & model override");
  await expect(
    page.getByRole("button", { name: "Validate problem", exact: true }),
  ).toBeDisabled();
  await page.screenshot({
    path: root + "/unsupported-preserved.png",
    fullPage: true,
  });
});
for (const dimension of [2, 3])
  test(`D Darcy ${dimension}D: forcing preview, static joint fields and compatible equation-driven routing`, async ({
    page,
  }) => {
    await define(page, "darcy", dimension);
    await expect(
      page.getByRole("table", { name: "Symbols in this problem" }),
    ).toContainText("Permeability");
    await expect(
      page.getByRole("table", { name: "Symbols in this problem" }),
    ).toContainText("Pressure");
    await page.getByLabel("Input preview field").selectOption("f");
    await expect(
      page.getByRole("img", { name: /actual known field/ }),
    ).toBeVisible();
    await expect(page.getByLabel("Input physical time")).toHaveCount(0);
    if (dimension === 3) {
      await page.getByLabel("Input slice normal").selectOption("0");
      await page.getByLabel("Input slice index").fill("3");
      await expect(page.getByText(/normal coordinate/)).toBeVisible();
    }
    await page.getByRole("button", { name: "Show LaTeX source" }).click();
    await page.getByLabel("LaTeX source 1").fill("-div(a grad(u))-f=0");
    await expect(
      page.getByText("checkpoint: yes", { exact: true }),
    ).toBeVisible();
    await information(page);
    await configure(page);
    await run(page);
    await expect(page.getByLabel("Physical time selection")).toHaveCount(0);
    await page.getByLabel("Physical field", { exact: true }).selectOption("1");
    if (dimension === 3) {
      await page.getByLabel("Slice normal", { exact: true }).selectOption("0");
      await page.getByLabel("Slice index", { exact: true }).fill("3");
      await expect(
        page.getByRole("img", { name: /Reconstruction, y \/ z, static field/ }),
      ).toBeVisible();
    }
    await page.screenshot({
      path: root + `/darcy-${dimension}d.png`,
      fullPage: true,
    });
  });
test("E chemistry: species meanings and one-species observations with physical time", async ({
  page,
}) => {
  await define(page, "chemistry");
  await expect(
    page.getByRole("table", { name: "Symbols in this problem" }),
  ).toContainText("Concentration of A");
  await page.getByLabel("Value k_f", { exact: true }).fill(".8");
  await information(page);
  await page.getByLabel("Input preview field").selectOption("c");
  await page.getByLabel("Input physical time").selectOption("2");
  await expect(page.getByText(/physical τ=/).first()).toBeVisible();
  await configure(page);
  await run(page);
  await page.getByLabel("Physical field", { exact: true }).selectOption("2");
  await page.getByLabel("Physical time selection").fill("2");
  await page.screenshot({ path: root + "/chemistry.png", fullPage: true });
});
test("F true 3D heat: real input and output coordinates and evaluation-only reference", async ({
  page,
}) => {
  await define(page);
  await page
    .getByRole("button", { name: "Change model / review layout" })
    .click();
  const models = await (await page.request.get("/api/models")).json();
  const model = models.find(
    (m: { specialist: string; problem: { grid: { shape: number[] } } }) =>
      m.specialist === "heat" && m.problem.grid.shape.length === 3,
  );
  await page.getByLabel("Registered checkpoint").selectOption(model.id);
  await page.getByRole("button", { name: "Use checkpoint layout" }).click();
  await page.getByRole("button", { name: "Apply reviewed changes" }).click();
  await expect(
    page.getByText("checkpoint: yes", { exact: true }),
  ).toBeVisible();
  await page.getByLabel("Input slice normal").selectOption("1");
  await information(page);
  await configure(page);
  await run(page);
  await page.getByLabel("Slice normal", { exact: true }).selectOption("0");
  await page.getByLabel("Slice index", { exact: true }).fill("3");
  await page.getByLabel("Physical time selection").fill("2");
  await page
    .getByRole("button", { name: "Attach synthetic reference for evaluation" })
    .click();
  await expect(
    page.getByRole("img", { name: /Absolute error, y \/ z/ }),
  ).toBeVisible();
  await page.screenshot({ path: root + "/heat-3d.png", fullPage: true });
});
test("G real FK: weighted predictions, actual ESS, ancestry and resampling events", async ({
  page,
}) => {
  await define(page);
  await information(page);
  await configure(page, "6");
  await page.getByLabel("Guidance mode").selectOption("fk");
  await page.getByLabel("Sample count", { exact: true }).fill("8");
  await expand(page, "FK particle & resampling settings");
  await page.getByLabel("FK ess_fraction").fill("1");
  await page.getByLabel("FK resample_every").fill("1");
  await run(page);
  await expect(page.getByLabel("FK terminal status")).toHaveText(
    "Terminal probability path completed",
  );
  await expect(page.getByText(/Weighted population mean/)).toBeVisible();
  await expect(
    page.getByRole("table", { name: "FK population trace" }),
  ).toContainText("scheduled_ESS");
  await expand(page, "Physical verification, process status & actions");
  await expect(
    page.getByRole("table", { name: "Execution decisions" }),
  ).toContainText("RESAMPLE");
  await page.screenshot({ path: root + "/fk.png", fullPage: true });
});
test("H actual managed training curves, checkpoints and discovered model metadata", async ({
  page,
}) => {
  await page.goto("/");
  await page
    .getByRole("button", { name: "Models & Training", exact: true })
    .click();
  await expect(
    page.getByRole("img", { name: /Training and validation loss/ }),
  ).toBeVisible();
  await expect(page.getByLabel("Training history run")).toContainText(
    "workflow-journey",
  );
  await expect(
    page.getByText(/generative training loss, not reconstruction error/),
  ).toBeVisible();
  await page.screenshot({
    path: root + "/training-history.png",
    fullPage: true,
  });
});
test("I history survives refresh; edited inputs retain prior result with an explicit stale banner", async ({
  page,
}) => {
  await define(page);
  await information(page);
  await configure(page);
  await run(page);
  await page.reload();
  await expect(
    page.getByRole("heading", { name: "Inspect the reconstruction" }),
  ).toBeVisible();
  await page.getByRole("button", { name: /1\s*Define/ }).click();
  await page.getByLabel("Diffusivity", { exact: true }).fill(".091");
  await page.getByRole("button", { name: /3\s*Reconstruct/ }).click();
  await expect(page.getByText(/Inputs changed since this run/)).toBeVisible();
  await expect(
    page.getByRole("img", { name: /Reconstruction,/ }),
  ).toBeVisible();
  await page.getByRole("button", { name: "Runs", exact: true }).click();
  await expect(
    page.getByRole("button", { name: "Open run", exact: true }).first(),
  ).toBeVisible();
});
test("J invalid observations, wrong-family overrides and real cancellation remain honest", async ({
  page,
}) => {
  await define(page);
  await information(page);
  await page.getByLabel("Import observations").setInputFiles({
    name: "bad.csv",
    mimeType: "text/csv",
    buffer: Buffer.from("field,x,value\nu,0,1"),
  });
  await expect(
    page.getByRole("alert").filter({ hasText: "Action needs attention" }),
  ).toContainText("required columns");
  await page.getByRole("button", { name: "Dismiss", exact: true }).click();
  await page.getByRole("button", { name: /1\s*Define/ }).click();
  await page
    .getByRole("button", { name: "Change model / review layout" })
    .click();
  const models = await (await page.request.get("/api/models")).json();
  const darcy = models.find(
    (m: { specialist: string }) => m.specialist === "darcy",
  );
  await page.getByLabel("Registered checkpoint").selectOption(darcy.id);
  await expect(
    page.getByText(/Explicit checkpoint override is incompatible/).first(),
  ).toBeVisible();
  await page.getByLabel("Registered checkpoint").selectOption("");
  await expect(
    page.getByText("checkpoint: yes", { exact: true }),
  ).toBeVisible();
  await page.getByRole("button", { name: /3\s*Reconstruct/ }).click();
  await expand(page, "Numerical sampling settings");
  await page.getByLabel("Generation steps", { exact: true }).fill("512");
  await page.getByRole("button", { name: "Validate complete request" }).click();
  await page
    .getByRole("button", { name: "Run reconstruction", exact: true })
    .click();
  await page.getByRole("button", { name: "Cancel job", exact: true }).click();
  await expect(page.getByText("cancelled · cancelled")).toBeVisible();
});
test("K compatible quality/cost comparison uses saved numerical jobs and rejects mismatched tasks", async ({
  page,
}) => {
  await page.goto("/");
  const jobs = await (await page.request.get("/api/jobs")).json();
  const groups = new Map<string, string[]>();
  let heat: string | undefined, darcy: string | undefined;
  for (const job of jobs.filter(
    (j: { status: string }) => j.status === "succeeded",
  )) {
    const result = await (
      await page.request.get(`/api/jobs/${job.id}/result`)
    ).json();
    const key = JSON.stringify([
      result.request.problem,
      result.request.observations,
    ]);
    groups.set(key, [...(groups.get(key) ?? []), job.id]);
    if (result.request.problem.family === "heat") heat = job.id;
    if (result.request.problem.family === "darcy") darcy = job.id;
  }
  const same = [...groups.values()].find((g) => g.length >= 2);
  expect(same).toBeTruthy();
  await page.getByRole("button", { name: "Runs", exact: true }).click();
  for (const id of same!.slice(0, 2))
    await page
      .getByLabel(`Compare run ${id.slice(0, 8)}`, { exact: true })
      .check();
  await page.getByRole("button", { name: "Compare selected runs" }).click();
  await expect(
    page.getByRole("img", {
      name: "Compatible runs: observation RMSE versus inference seconds",
    }),
  ).toBeVisible();
  await expect(
    page.getByRole("table", { name: "Compatible run comparison" }),
  ).toContainText("Unavailable");
  await page.screenshot({ path: root + "/comparison.png", fullPage: true });
  for (const id of same!.slice(0, 2))
    await page
      .getByLabel(`Compare run ${id.slice(0, 8)}`, { exact: true })
      .uncheck();
  expect(heat && darcy).toBeTruthy();
  for (const id of [heat!, darcy!])
    await page
      .getByLabel(`Compare run ${id.slice(0, 8)}`, { exact: true })
      .check();
  await page.getByRole("button", { name: "Compare selected runs" }).click();
  await expect(page.getByRole("alert")).toContainText(
    "Direct quality/cost comparison requires",
  );
});
