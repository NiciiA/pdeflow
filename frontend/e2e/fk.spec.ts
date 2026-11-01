import { experts, physical } from "./disclosure";
import { test, expect } from "@playwright/test";
for (const dim of [1, 2, 3])
  test(`real FK ${dim}D weighted reconstruction, ancestry and detached reference`, async ({
    page,
  }) => {
    await page.goto("/");
    await page
      .getByRole("button", { name: "Reset draft", exact: true })
      .click();
    await experts(page);
    await experts(page);
    await page.getByRole("button", { name: /^Heat \/ diffusion/ }).click();
    await experts(page);
    if (dim !== 2) {
      const models = await (await page.request.get("/api/models")).json();
      const model = models.find(
        (m: { specialist: string; problem: { grid: { shape: number[] } } }) =>
          m.specialist === "heat" && m.problem.grid.shape.length === dim,
      );
      await page.getByLabel("Registered checkpoint").selectOption(model.id);
      await page.getByRole("button", { name: "Use checkpoint layout" }).click();
      await experts(page);
      await page
        .getByRole("button", { name: "Apply reviewed changes" })
        .click();
      await experts(page);
    }
    await expect(page.getByText("checkpoint: yes")).toBeVisible();
    await page
      .getByRole("button", { name: "Continue to observations" })
      .click();
    await experts(page);
    await page
      .getByRole("button", { name: "Generate synthetic observations" })
      .click();
    await experts(page);
    await page
      .getByRole("button", { name: "Configure reconstruction" })
      .click();
    await experts(page);
    await page.getByLabel("Guidance mode").selectOption("fk");
    await experts(page);
    await expect(
      page.getByText(/not physical molecules/).first(),
    ).toBeVisible();
    await page.getByLabel("Sample count", { exact: true }).fill("8");
    await page.getByLabel("Generation steps", { exact: true }).fill("6");
    await page.getByLabel("FK ess_fraction").fill("1");
    await page.getByLabel("FK resample_every").fill("1");
    await page.getByLabel("FK observation_weight").fill("1");
    await page
      .getByRole("button", { name: "Validate complete request" })
      .click();
    await experts(page);
    await expect(page.getByText("Compatible request")).toBeVisible();
    await page
      .getByRole("button", { name: "Run reconstruction", exact: true })
      .click();
    await experts(page);
    await expect(page.getByLabel("FK terminal status")).toHaveText(
      "Terminal probability path completed",
      { timeout: 30000 },
    );
    await expect(
      page.getByRole("table", { name: "FK population trace" }),
    ).toContainText("scheduled_ESS");
    await physical(page);
    await expect(
      page.getByRole("table", { name: "Execution decisions" }),
    ).toContainText("RESAMPLE");
    await expect(
      page.getByText("No evaluation reference attached", { exact: true }),
    ).toBeVisible();
    if (dim === 3) {
      await page.getByLabel("Slice normal").selectOption("0");
      await page.getByLabel("Slice index").fill("3");
    }
    await page
      .getByRole("button", {
        name: "Attach synthetic reference for evaluation",
      })
      .click();
    await experts(page);
    await expect(
      page.getByRole("table", {
        name: "Per-field reconstruction errors",
        exact: true,
      }),
    ).toBeVisible();
    await page
      .getByRole("button", { name: "Remove evaluation reference" })
      .click();
    await experts(page);
    await expect(
      page.getByRole("table", {
        name: "Per-field reconstruction errors",
        exact: true,
      }),
    ).toHaveCount(0);
    await page.screenshot({
      path: `../outputs/workbench-redesign-20261025/browser/regression/fk-${dim}d.png`,
      fullPage: true,
    });
    await page.reload();
    await page.getByRole("button", { name: /^Runs$/ }).click();
    await experts(page);
    await page
      .getByRole("button", { name: "Open run", exact: true })
      .first()
      .click();
    await experts(page);
    await expect(page.getByLabel("FK terminal status")).toHaveText(
      "Terminal probability path completed",
    );
  });

test("real FK rejects incompatible controls and cancellation is not terminal completion", async ({
  page,
}) => {
  await page.goto("/");
  await page.getByRole("button", { name: "Reset draft", exact: true }).click();
  await experts(page);
  await experts(page);
  await page.getByRole("button", { name: /^Heat \/ diffusion/ }).click();
  await experts(page);
  await expect(page.getByText("checkpoint: yes")).toBeVisible();
  await page.getByRole("button", { name: "Continue to observations" }).click();
  await experts(page);
  await page
    .getByRole("button", { name: "Generate synthetic observations" })
    .click();
  await experts(page);
  await page.getByRole("button", { name: "Configure reconstruction" }).click();
  await experts(page);
  await page.getByLabel("Guidance mode").selectOption("fk");
  await experts(page);
  await page.getByLabel("Sample count", { exact: true }).fill("8");
  await page
    .getByText("Feedback controller and budgets", { exact: true })
    .click();
  await experts(page);
  await page
    .getByLabel("Process configuration JSON")
    .fill(JSON.stringify({ refine: true, max_attempts: 2 }));
  await page
    .getByRole("button", { name: "Apply process configuration" })
    .click();
  await experts(page);
  await page.getByRole("button", { name: "Validate complete request" }).click();
  await experts(page);
  await expect(
    page.getByRole("alert").filter({ hasText: "Action needs attention" }),
  ).toContainText("FK plus refinement/retry is unsupported");
  await page.getByRole("button", { name: "Dismiss" }).click();
  await experts(page);
  await page.getByLabel("Process configuration JSON").fill(
    JSON.stringify({
      max_sampling_steps: 512,
      max_network_evaluations: 1024,
      max_particles: 8,
    }),
  );
  await page
    .getByRole("button", { name: "Apply process configuration" })
    .click();
  await experts(page);
  await page.getByLabel("FK ess_fraction").fill("0");
  await page.getByRole("button", { name: "Validate complete request" }).click();
  await experts(page);
  await expect(
    page.getByRole("alert").filter({ hasText: "Action needs attention" }),
  ).toContainText("ESS fraction");
  await page.getByRole("button", { name: "Dismiss" }).click();
  await experts(page);
  await page.getByLabel("FK ess_fraction").fill(".7");
  await page.getByLabel("Generation steps", { exact: true }).fill("512");
  await page.getByRole("button", { name: "Validate complete request" }).click();
  await experts(page);
  await page
    .getByRole("button", { name: "Run reconstruction", exact: true })
    .click();
  await experts(page);
  await page.getByRole("button", { name: "Cancel job", exact: true }).click();
  await experts(page);
  await expect(page.getByText(/STOP_CANCELLED/).first()).toBeVisible();
  await expect(page.getByLabel("FK terminal status")).toHaveCount(0);
});
