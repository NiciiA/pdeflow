import { test, expect } from "@playwright/test";
import type { Page } from "@playwright/test";
const root = process.env.PDEFLOW_BROWSER_OUTPUT ?? "../outputs/flow-workspace-20261026/browser";
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
async function reference(page: Page) {
  await page.getByRole("button", { name: /2\s*Add information/ }).click();
  await page
    .getByRole("button", {
      name: "Load visible reference defaults",
      exact: true,
    })
    .click();
  await page
    .getByRole("button", { name: "Solve reference", exact: true })
    .click();
  await expect(page.getByText(/Immutable reference/)).toBeVisible();
  await page.getByLabel("Sampling count", { exact: true }).fill("3");
  await page
    .getByRole("button", { name: "Generate observations", exact: true })
    .click();
  await expect(page.getByText(/noisy measurements\. Clean/)).toBeVisible();
  await page
    .getByRole("button", { name: "Reconstruct from measurements", exact: true })
    .click();
}
async function configure(page: Page, steps = "4") {
  await expand(page, "Numerical sampling settings");
  await page.getByLabel("Generation steps", { exact: true }).fill(steps);
  await page.getByLabel("Sample count", { exact: true }).fill("2");
  await page
    .getByRole("button", { name: "Validate complete request", exact: true })
    .click();
  await expect(page.getByText("Compatible request")).toBeVisible();
}
for (const [family, dimension] of [
  ["heat", 2],
  ["darcy", 3],
  ["chemistry", 2],
] as const) {
  test(`real ${family} ${dimension}D linked field, automatic comparison, generation and reopen`, async ({
    page,
  }) => {
    test.setTimeout(120000);
    const errors: string[] = [];
    page.on("pageerror", (e) => errors.push(e.message));
    await define(page, family, dimension);
    await reference(page);
    await configure(page);
    let submissions = 0;
    page.on("request", (r) => {
      if (r.method() === "POST" && r.url().endsWith("/api/jobs")) {
        submissions++;
        expect(r.postDataJSON().model_selection.mode).toBe("auto");
        expect(r.postDataJSON()).not.toHaveProperty("model_id");
        expect(r.postDataJSON()).not.toHaveProperty("reference_id");
      }
    });
    await page
      .getByRole("button", { name: "Run reconstruction", exact: true })
      .click();
    const area = page.getByRole("region", {
      name: "Linked mathematical workspace",
    });
    await expect(
      area.getByRole("img", { name: /Reference, linked/ }).first(),
    ).toBeVisible({ timeout: 90000 });
    expect(submissions).toBe(1);
    await page.getByLabel("Inspection x index", { exact: true }).fill("2");
    await page.getByLabel("Lock selected point", { exact: true }).check();
    await expect(area.getByText(/x=0.25000|x=0.31250/)).toBeVisible();
    await page.screenshot({
      path: `${root}/${family}-${dimension}d-spatial.png`,
      fullPage: true,
    });
    await area.screenshot({
      path: `${root}/${family}-${dimension}d-linked-focus.png`,
    });
    if (family === "heat") {
      await page.setViewportSize({ width: 430, height: 900 });
      await area.screenshot({ path: root + "/linked-narrow.png" });
      expect(
        await page.evaluate(() => document.documentElement.scrollWidth),
      ).toBeLessThanOrEqual(431);
      await page.setViewportSize({ width: 1440, height: 1000 });
    }
    if (family === "darcy") {
      await page
        .getByRole("button", { name: "Physical evolution", exact: true })
        .click();
      await expect(
        area.getByText(/Static Darcy has no physical-time/),
      ).toBeVisible();
      await expect(
        page.getByLabel("Physical evolution time", { exact: true }),
      ).toHaveCount(0);
    } else {
      await page
        .getByRole("button", { name: "Physical evolution", exact: true })
        .click();
      await page
        .getByLabel("Physical evolution time", { exact: true })
        .fill("1");
      await expect(area.getByText("τ=0.05", { exact: true })).toBeVisible();
    }
    await page
      .getByRole("button", { name: "Generative evolution", exact: true })
      .click();
    await expect(
      page.getByLabel("Generation progress s", { exact: true }),
    ).toBeVisible();
    await page.getByLabel("Generation progress s", { exact: true }).fill("2");
    await expect(
      page.getByRole("img", { name: /Current noisy\/generated field/ }),
    ).toBeVisible();
    await page
      .getByLabel("Generative quantity", { exact: true })
      .selectOption("endpoint_physical");
    await expect(
      page.getByRole("img", { name: /Estimated final field/ }),
    ).toBeVisible();
    await page
      .getByRole("button", { name: "Play generation", exact: true })
      .click();
    await page
      .getByRole("button", { name: "Pause generation", exact: true })
      .click();
    if (family === "chemistry")
      await page
        .getByLabel("Generation field", { exact: true })
        .selectOption("1");
    if (dimension === 3) {
      await page
        .getByLabel("Generation slice normal", { exact: true })
        .selectOption("0");
      await page
        .getByLabel("Generation slice index", { exact: true })
        .fill("3");
    }
    await page.screenshot({
      path: `${root}/${family}-${dimension}d-generation.png`,
      fullPage: true,
    });
    await page.reload();
    await expect(
      page.getByRole("table", {
        name: "Reference comparison errors",
        exact: true,
      }),
    ).toBeVisible({ timeout: 30000 });
    await expect(
      page.getByRole("button", { name: "Compare", exact: true }),
    ).toBeVisible();
    await page
      .getByRole("button", { name: "Clear workspace", exact: true })
      .click();
    await page
      .getByRole("button", { name: "Clear current draft", exact: true })
      .click();
    await expect(
      page.getByRole("heading", {
        name: "Define the physical problem",
        exact: true,
      }),
    ).toBeVisible();
    await page.reload();
    await expect(
      page.locator('math-field[aria-label="Equation 1"]'),
    ).toHaveJSProperty("value", "");
    await expect(
      page.getByRole("button", { name: "Heat / diffusion", exact: false }),
    ).toBeVisible();
    await page.getByRole("button", { name: "Runs", exact: true }).click();
    await expect(
      page.getByRole("cell", { name: "succeeded", exact: true }).first(),
    ).toBeVisible();
    await page
      .getByRole("button", { name: "Models & Training", exact: true })
      .click();
    await expect(page.getByText(/parameters · width/).first()).toBeVisible();
    expect(errors).toEqual([]);
  });
}

test("slow registry and late parse cannot serialize empty selection or restore a cleared draft", async ({
  page,
}) => {
  await page.route("**/api/models", async (route) => {
    await new Promise((r) => setTimeout(r, 600));
    await route.continue();
  });
  await define(page);
  await page.route("**/api/mathematics/parse", async (route) => {
    await new Promise((r) => setTimeout(r, 700));
    await route.continue();
  });
  await page.getByLabel("Diffusivity", { exact: true }).fill(".095");
  await page
    .getByRole("button", { name: "Clear workspace", exact: true })
    .click();
  await page
    .getByRole("button", { name: "Clear current draft", exact: true })
    .click();
  await page.waitForTimeout(1100);
  await expect(
    page.locator('math-field[aria-label="Equation 1"]'),
  ).toHaveJSProperty("value", "");
  await page.reload();
  await expect(
    page.locator('math-field[aria-label="Equation 1"]'),
  ).toHaveJSProperty("value", "");
});

test("clear while running offers keep/cancel/return and never repopulates the editor", async ({
  page,
}) => {
  test.setTimeout(90000);
  await define(page);
  await reference(page);
  await configure(page, "256");
  await page
    .getByRole("button", { name: "Run reconstruction", exact: true })
    .click();
  await expect(
    page.getByRole("button", { name: "Cancel job", exact: true }),
  ).toBeVisible();
  await page
    .getByRole("button", { name: "Clear workspace", exact: true })
    .click();
  await expect(
    page.getByRole("button", {
      name: "Keep job running and clear editor",
      exact: true,
    }),
  ).toBeVisible();
  await page
    .getByRole("button", { name: "Return without clearing", exact: true })
    .click();
  await page
    .getByRole("button", { name: "Clear workspace", exact: true })
    .click();
  await page
    .getByRole("button", { name: "Cancel job, then clear", exact: true })
    .click();
  await expect(
    page.locator('math-field[aria-label="Equation 1"]'),
  ).toHaveJSProperty("value", "");
  await page.waitForTimeout(700);
  await page.reload();
  await expect(
    page.locator('math-field[aria-label="Equation 1"]'),
  ).toHaveJSProperty("value", "");
});

test("FK playback shows actual ancestry changes and bounded missing-parent gaps", async ({
  page,
  request,
}) => {
  test.setTimeout(90000);
  const models = await (await request.get("/api/models")).json();
  const model = models.find(
    (m: { problem: { family: string; grid: { shape: number[] } } }) =>
      m.problem.family === "heat" && m.problem.grid.shape.join(",") === "8,8",
  );
  const demo = await (
    await request.post("/api/demos", {
      data: { model_id: model.id, sensors: 4 },
    })
  ).json();
  const submitted = await request.post("/api/jobs", {
    data: {
      model_selection: { mode: "auto" },
      problem: demo.problem,
      rows: demo.rows,
      mode: "fk",
      inference: {
        steps: 4,
        particles: 8,
        controller: { max_particles: 8 },
        fk: {
          chunk_size: 4,
          ess_fraction: 1,
          resample_every: 1,
          observation_weight: 0.01,
          physics_weight: 0.0001,
        },
        snapshots: { max_frames: 5, max_particles: 4 },
      },
    },
  });
  expect(submitted.status()).toBe(202);
  const job = await submitted.json();
  await expect
    .poll(
      async () =>
        (await (await request.get("/api/jobs/" + job.id)).json()).status,
      { timeout: 60000 },
    )
    .toBe("succeeded");
  const meta = await (
    await request.get("/api/jobs/" + job.id + "/generation")
  ).json();
  let chosen = { frame: -1, particle: 0 };
  for (let f = 1; f < meta.frames.length; f++) {
    const p = meta.frames[f].parents_at_previous_snapshot.findIndex(
      (parent: number, i: number) => parent !== i,
    );
    if (p >= 0) {
      chosen = { frame: f, particle: p };
      break;
    }
  }
  expect(chosen.frame).toBeGreaterThan(0);
  await page.addInitScript((id) => {
    localStorage.setItem("pdeflow.selected-job.v1", id);
    localStorage.setItem("pdeflow.view.v1", "Results");
  }, job.id);
  await page.goto("/");
  await page
    .getByRole("button", { name: "Generative evolution", exact: true })
    .click();
  await page
    .getByLabel("Generation progress s", { exact: true })
    .fill(String(chosen.frame));
  await page
    .getByLabel("Generation particle", { exact: true })
    .selectOption(String(chosen.particle));
  await expect(page.getByText(/Ancestry transition:/)).toBeVisible();
  await page.screenshot({ path: root + "/fk-ancestry.png", fullPage: true });
});

test("keep a running job while clearing leaves the editor empty after job completion", async ({
  page,
  request,
}) => {
  test.setTimeout(90000);
  await define(page);
  await reference(page);
  await configure(page, "12");
  const response = page.waitForResponse(
    (r) => r.url().endsWith("/api/jobs") && r.request().method() === "POST",
  );
  await page
    .getByRole("button", { name: "Run reconstruction", exact: true })
    .click();
  const job = await (await response).json();
  await page
    .getByRole("button", { name: "Clear workspace", exact: true })
    .click();
  await page
    .getByRole("button", {
      name: "Keep job running and clear editor",
      exact: true,
    })
    .click();
  await expect(
    page.locator('math-field[aria-label="Equation 1"]'),
  ).toHaveJSProperty("value", "");
  await expect
    .poll(
      async () =>
        (await (await request.get("/api/jobs/" + job.id)).json()).status,
      { timeout: 60000 },
    )
    .toBe("succeeded");
  await expect(
    page.locator('math-field[aria-label="Equation 1"]'),
  ).toHaveJSProperty("value", "");
  await page.reload();
  await expect(
    page.locator('math-field[aria-label="Equation 1"]'),
  ).toHaveJSProperty("value", "");
});

test("1024² reference uses bounded display and preserves the native requested grid", async ({
  page,
}) => {
  test.setTimeout(90000);
  await define(page);
  await expand(page, "Edit declarations, domain, conditions & model override");
  await page.getByLabel("x nodes", { exact: true }).fill("1024");
  await page.getByLabel("y nodes", { exact: true }).fill("1024");
  await expect(
    page.getByText("checkpoint: unavailable", { exact: true }),
  ).toBeVisible();
  await page.getByRole("button", { name: /2\s*Add information/ }).click();
  await page
    .getByRole("button", {
      name: "Load visible reference defaults",
      exact: true,
    })
    .click();
  const preview = page.waitForResponse((r) =>
    r.url().includes("/slice?field_index=0"),
  );
  await page
    .getByRole("button", { name: "Solve reference", exact: true })
    .click();
  const r = await preview;
  const v = await r.json();
  expect(v.native_shape).toEqual([1024, 1024]);
  expect(v.shape).toEqual([128, 128]);
  expect(v.display_downsampled).toBeTruthy();
  expect((await r.body()).length).toBeLessThan(1000000);
  await expect(
    page.getByText(
      /Display 128×128 sampled from native numerical grid 1024×1024/,
    ),
  ).toBeVisible();
  await page.screenshot({
    path: root + "/native-1024-reference.png",
    fullPage: true,
  });
});

test("stale explicit checkpoint produces actionable error without losing measurements", async ({
  page,
  request,
}) => {
  const ref = await (
    await request.post("/api/references", {
      data: JSON.parse(
        (await import("node:fs")).readFileSync(
          "../configs/reference/heat-2d.json",
          "utf8",
        ),
      ),
    })
  ).json();
  const observed = await (
    await request.post("/api/references/" + ref.id + "/sample", {
      data: { sampling: { count: 3 } },
    })
  ).json();
  await page.addInitScript((v) => {
    localStorage.setItem(
      "pdeflow.draft.v2",
      JSON.stringify({
        version: 2,
        math: v.mathematical_request,
        modelOverride: "0".repeat(32),
        inference: { steps: 2, particles: 1 },
      }),
    );
    localStorage.setItem("pdeflow.view.v1", "Reconstruction");
  }, observed);
  await page.goto("/");
  await page
    .getByRole("button", { name: "Validate complete request", exact: true })
    .click();
  await expect(
    page.getByRole("alert").filter({ hasText: "checkpoint is unavailable" }),
  ).toBeVisible();
  await expect(
    page.getByText("12 measurements · grid 8 × 8 · 4 physical times", {
      exact: true,
    }),
  ).toBeVisible();
  await expect(
    page.getByRole("button", { name: "Run reconstruction", exact: true }),
  ).toBeDisabled();
});

test("reference-free manual observations retain physical diagnostics and actual generation", async ({
  page,
}) => {
  await define(page);
  await page.getByRole("button", { name: /2\s*Add information/ }).click();
  await page
    .getByLabel("Import observations", { exact: true })
    .setInputFiles({
      name: "manual.csv",
      mimeType: "text/csv",
      buffer: Buffer.from(
        "field,x,y,time,value\nu,0,0,0,0.8\nu,0.25,0.5,0.05,0.3\n",
      ),
    });
  await page
    .getByRole("button", { name: "Configure reconstruction", exact: true })
    .click();
  await configure(page, "3");
  await page
    .getByRole("button", { name: "Run reconstruction", exact: true })
    .click();
  const area = page.getByRole("region", {
    name: "Linked mathematical workspace",
  });
  await expect(
    area.getByRole("img", { name: /Reconstruction, linked/ }),
  ).toBeVisible({ timeout: 60000 });
  await expect(area.getByText(/No reference is attached/)).toBeVisible();
  await expect(
    page.getByText("No evaluation reference attached", { exact: true }),
  ).toBeVisible();
  await page
    .getByRole("button", { name: "Generative evolution", exact: true })
    .click();
  await expect(
    page.getByLabel("Generation progress s", { exact: true }),
  ).toBeVisible();
});

test("valid explicit registered selection is serialized explicitly and independently revalidated", async ({
  page,
  request,
}) => {
  await define(page);
  const models = await (await request.get("/api/models")).json();
  const m = models.find(
    (m: { problem: { family: string; grid: { shape: number[] } } }) =>
      m.problem.family === "heat" && m.problem.grid.shape.join(",") === "8,8",
  );
  await expand(page, "Edit declarations, domain, conditions & model override");
  await page
    .getByLabel("Registered checkpoint", { exact: true })
    .selectOption(m.id);
  await reference(page);
  const resolved = page.waitForResponse((r) =>
    r.url().endsWith("/api/reconstruction/validate"),
  );
  await configure(page, "2");
  const r = await resolved;
  expect(r.request().postDataJSON().model_selection).toEqual({
    mode: "explicit",
    id: m.id,
  });
  expect((await r.json()).resolved.model_id).toBe(m.id);
});
