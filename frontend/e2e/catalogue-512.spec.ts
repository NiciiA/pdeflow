import { test, expect } from "@playwright/test";
const root = "../outputs/catalogue-512-20261026/browser";
async function expand(page: import("@playwright/test").Page, name: string) {
  const exact = page.locator("summary").filter({ hasText: name });
  const d = exact.locator("..");
  if (!(await d.getAttribute("open"))) await exact.click();
}
test("chemistry mismatch, current catalogue plan, actual history and preview-only storage", async ({
  page,
  request,
}) => {
  test.setTimeout(120000);
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(e.message));
  let jobs = 0;
  page.on("request", (r) => {
    if (r.method() === "POST" && r.url().endsWith("/api/jobs")) jobs++;
  });
  await page.goto("/");
  await page.getByRole("button", { name: /^A\+B/ }).click();
  await expand(page, "Edit declarations, domain, conditions & model override");
  await page.getByLabel("x nodes", { exact: true }).fill("12");
  await page.getByLabel("y nodes", { exact: true }).fill("12");
  await expect(
    page.getByText("checkpoint: unavailable", { exact: true }),
  ).toBeVisible();
  await expand(page, "Checkpoint exclusions and restrictions");
  await expect(
    page.getByText('"grid.shape"', { exact: false }).first(),
  ).toBeVisible();
  await expect(
    page.getByText('"expected": [', { exact: false }).first(),
  ).toBeVisible();
  await page
    .getByText('\"grid.shape\"', { exact: false })
    .first()
    .scrollIntoViewIfNeeded();
  await page.screenshot({ path: root + "/chemistry-mismatch.png" });
  await page
    .getByRole("button", { name: "Models & Training", exact: true })
    .click();
  await expect(
    page.getByRole("table", { name: "Training coverage", exact: true }),
  ).toBeVisible();
  await page
    .getByRole("button", {
      name: "Plan full catalogue (no training)",
      exact: true,
    })
    .click();
  await expect(
    page.getByText(/11 sequential targets · no jobs launched/),
  ).toBeVisible();
  const managed = page
    .getByLabel("Live training run", { exact: true })
    .locator("option")
    .filter({ hasText: "local-pilot" })
    .first();
  await page
    .getByLabel("Live training run", { exact: true })
    .selectOption((await managed.getAttribute("value"))!);
  await page
    .getByText("Actual managed progress", { exact: true })
    .scrollIntoViewIfNeeded();
  await page.screenshot({ path: root + "/training-history.png" });
  await expect(page.getByText(/Latest resumable checkpoint/)).toBeVisible();
  await page
    .getByRole("button", { name: "Show cached storage", exact: true })
    .click();
  await expect(
    page.getByRole("table", { name: "Storage inventory", exact: true }),
  ).toBeVisible({ timeout: 60000 });
  await page
    .getByRole("button", { name: "Preview cleanup (no deletion)", exact: true })
    .click();
  await expect(page.getByText(/Nothing deleted or relocated/)).toBeVisible();
  await page.getByText(/Nothing deleted or relocated/).scrollIntoViewIfNeeded();
  await page.screenshot({ path: root + "/training-storage.png" });
  expect(jobs).toBe(0);
  expect(errors).toEqual([]);
  const coverage = await (
    await request.get("http://127.0.0.1:8000/api/training-coverage")
  ).json();
  expect(coverage.targets).toHaveLength(11);
  expect(
    coverage.targets.find((t: { id: string }) => t.id === "chemistry-2d").state,
  ).toBe("completed");
});
test("native 512 heat template, edited candidate verification, actual reconstruction and automatic comparison", async ({
  page,
}) => {
  test.setTimeout(180000);
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(e.message));
  await page.goto("/");
  await page.getByRole("button", { name: /^Heat \/ diffusion/ }).click();
  await page
    .getByRole("button", { name: "Load 512² heat example", exact: true })
    .click();
  await page
    .getByRole("button", { name: "Apply reviewed changes", exact: true })
    .click();
  await expect(
    page.getByText("checkpoint: yes", { exact: true }),
  ).toBeVisible();
  await page.getByRole("button", { name: /2\s*Add information/ }).click();
  await page
    .getByText("Analytical example · editable initial and complete reference", {
      exact: true,
    })
    .click();
  await page
    .getByRole("button", {
      name: "Replace reference inputs with reviewed example",
      exact: true,
    })
    .click();
  const candidate = page.getByLabel("Analytical reference expression", {
    exact: true,
  });
  const original = await candidate.inputValue();
  await candidate.fill("cos(2*pi*x)*cos(2*pi*y)");
  await page
    .getByRole("button", { name: "Verify analytical candidate", exact: true })
    .click();
  await expect(page.getByLabel("Analytical verification")).toContainText(
    "inconsistent user-supplied candidate",
  );
  await candidate.fill(original);
  await page
    .getByRole("button", { name: "Verify analytical candidate", exact: true })
    .click();
  await expect(page.getByLabel("Analytical verification")).toContainText(
    '"passed": true',
  );
  await page
    .getByRole("button", { name: "Solve reference", exact: true })
    .click();
  await expect(page.getByText(/Immutable reference/)).toBeVisible({
    timeout: 60000,
  });
  await page.getByLabel("Sampling count", { exact: true }).fill("3");
  await page
    .getByRole("button", { name: "Generate observations", exact: true })
    .click();
  await expect(page.getByText(/noisy measurements\. Clean/)).toBeVisible();
  await page
    .getByRole("button", { name: "Reconstruct from measurements", exact: true })
    .click();
  await expand(page, "Numerical sampling settings");
  await page.getByLabel("Generation steps", { exact: true }).fill("2");
  await page.getByLabel("Sample count", { exact: true }).fill("1");
  await expand(page, "Feedback controller and budgets");
  await page
    .getByRole("button", {
      name: "Use requested native grid budget (262144)",
      exact: true,
    })
    .click();
  await page
    .getByRole("button", { name: "Validate complete request", exact: true })
    .click();
  await expect(page.getByText("Compatible request")).toBeVisible();
  const posted = page.waitForRequest(
    (r) => r.method() === "POST" && r.url().endsWith("/api/jobs"),
  );
  await page
    .getByRole("button", { name: "Run reconstruction", exact: true })
    .click();
  const body = (await posted).postDataJSON();
  expect(body.problem.grid.shape).toEqual([512, 512]);
  expect(body.problem.grid.times).toEqual([0, 0.05, 0.1]);
  expect(JSON.stringify(body)).not.toContain("analytical_reference");
  const area = page.getByRole("region", {
    name: "Linked mathematical workspace",
  });
  await expect(
    area.getByRole("img", { name: /Reference, linked/ }).first(),
  ).toBeVisible({ timeout: 120000 });
  await area.screenshot({ path: root + "/native-512-linked.png" });
  await page
    .getByRole("button", { name: "Generative evolution", exact: true })
    .click();
  await expect(
    area.getByText("Generation progress s", { exact: true }),
  ).toBeVisible();
  await area.screenshot({ path: root + "/native-512-generation.png" });
  expect(errors).toEqual([]);
});
test("actual bounded CLI worker remains active independently of browser lifetime", async ({
  page,
  browser,
  request,
}) => {
  test.skip(
    process.env.PDEFLOW_MPS_BROWSER !== "1",
    "Explicit MacBook MPS integration; never launch native training in an ordinary browser test run",
  );
  test.setTimeout(90000);
  const { spawn } = await import("node:child_process");
  const { resolve } = await import("node:path");
  const { createWriteStream, writeFileSync, readFileSync } =
    await import("node:fs");
  const repo = resolve(".."),
    evidence = resolve(repo, "outputs/catalogue-512-20261026");
  await page.goto("/");
  await page
    .getByRole("button", { name: "Models & Training", exact: true })
    .click();
  await expect(
    page.getByRole("table", { name: "Training coverage", exact: true }),
  ).toBeVisible();
  const resumePath = process.env.PDEFLOW_MONITOR_RESUME;
  const saved = resumePath
    ? JSON.parse(readFileSync(resolve(repo, resumePath, "run.json"), "utf8"))
    : null;
  const additional = Number(process.env.PDEFLOW_MONITOR_ADDITIONAL ?? 0);
  if (!Number.isInteger(additional) || additional < 0 || additional > 2)
    throw new Error("Monitoring extension limited to two verification updates");
  const expectedBudget = saved ? saved.budget + additional : 8;
  const name = saved?.name ?? "browser-monitor-native512-eight-" + Date.now();
  const log = createWriteStream(resolve(evidence, "monitor-cli-final.log"));
  const child = spawn(
    resolve(repo, ".venv/bin/pdeflow"),
    resumePath
      ? [
          "resume",
          resolve(repo, resumePath),
          ...(additional ? ["--additional-steps", String(additional)] : []),
          "--workspace",
          resolve(evidence, "workspace"),
        ]
      : [
          "train",
          "--config",
          resolve(evidence, "native-512-bounded-config.json"),
          "--steps",
          "8",
          "--stop-after",
          "7",
          "--device",
          "mps",
          "--workspace",
          resolve(evidence, "workspace"),
          "--name",
          name,
        ],
    { cwd: repo, stdio: ["ignore", "pipe", "pipe"] },
  );
  child.stdout.pipe(log);
  child.stderr.pipe(log);
  const finished = new Promise<number | null>((ok, reject) => {
    child.once("error", reject);
    child.once("close", (code) => {
      log.end();
      ok(code);
    });
  });
  try {
    await expect
      .poll(
        async () => {
          const runs = await (
            await request.get("http://127.0.0.1:8000/api/training-runs")
          ).json();
          return runs.some(
            (r: { name: string; active: boolean }) =>
              r.name === name && r.active,
          );
        },
        { timeout: 20000 },
      )
      .toBe(true);
    await page
      .getByRole("button", {
        name: "Refresh training coverage & progress",
        exact: true,
      })
      .click();
    await expect(page.getByText(/1 active worker\(s\)/)).toBeVisible({
      timeout: 15000,
    });
    await expect
      .poll(
        async () => {
          const runs = await (
            await request.get("http://127.0.0.1:8000/api/training-runs")
          ).json();
          const r = runs.find((r: { name: string }) => r.name === name);
          return r?.live?.step ?? 0;
        },
        { timeout: 20000 },
      )
      .toBeGreaterThan(0);
    await page
      .getByText("Actual managed progress", { exact: true })
      .scrollIntoViewIfNeeded();
    await page.screenshot({ path: root + "/active-cli-progress.png" });
    expect(child.exitCode).toBeNull();
    await page.close();
    expect(await finished).toBe(0);
    if (!resumePath) {
      const append = createWriteStream(
        resolve(evidence, "monitor-cli-final.log"),
        { flags: "a" },
      );
      const resumed = spawn(
        resolve(repo, ".venv/bin/pdeflow"),
        ["resume", "latest", "--workspace", resolve(evidence, "workspace")],
        { cwd: repo, stdio: ["ignore", "pipe", "pipe"] },
      );
      resumed.stdout.pipe(append);
      resumed.stderr.pipe(append);
      expect(
        await new Promise<number | null>((ok, reject) => {
          resumed.once("error", reject);
          resumed.once("close", (code) => {
            append.end();
            ok(code);
          });
        }),
      ).toBe(0);
    }
    const reopened = await browser.newPage();
    await reopened.goto("http://127.0.0.1:5173");
    await reopened
      .getByRole("button", { name: "Models & Training", exact: true })
      .click();
    await expect(
      reopened
        .getByLabel("Live training run", { exact: true })
        .locator("option")
        .filter({ hasText: name }),
    ).toHaveText(new RegExp(name + ".*completed"));
    const runs = await (
      await request.get("http://127.0.0.1:8000/api/training-runs")
    ).json();
    const r = runs.find((r: { name: string }) => r.name === name);
    expect(r.progress).toBe(expectedBudget);
    expect(r.active).toBe(false);
    writeFileSync(
      resolve(evidence, "monitor-browser.json"),
      JSON.stringify(
        {
          run_id: r.run_id,
          progress: r.progress,
          budget: r.budget,
          active_after_exit: r.active,
          browser_closed_before_cli_finished: true,
          device: r.device,
          live: r.live,
        },
        null,
        2,
      ),
    );
    await reopened.close();
  } finally {
    if (child.exitCode === null) {
      child.kill("SIGINT");
      await finished;
    }
  }
});
