import { test, expect } from "@playwright/test";
import fs from "node:fs";
import path from "node:path";
test.use({
  baseURL: process.env.PDEFLOW_LOSS_BROWSER_URL || "http://127.0.0.1:5173",
});
const evidence = path.resolve("../outputs/standard-mse-20261101/browser");
test("real new-run selection reaches saved A1/A2 objectives", async ({
  page,
}) => {
  fs.mkdirSync(evidence, { recursive: true });
  await page.goto("/");
  await page
    .getByRole("navigation", { name: "Application" })
    .getByRole("button", { name: "Train model", exact: true })
    .click();
  await page
    .getByLabel("Training budget", { exact: true })
    .selectOption("pipeline-check");
  await page.getByLabel("Training device", { exact: true }).selectOption("cpu");
  const loss = page.getByLabel("Field reconstruction loss", { exact: true });
  await expect(loss).toHaveValue("mse");
  await expect(
    page.getByLabel("Feature balance: α", { exact: true }),
  ).toHaveCount(0);
  await expect(
    page.getByText("Advanced region analysis", { exact: true }),
  ).toHaveCount(0);
  const records = [];
  for (const type of ["mse", "region_balanced_mse"]) {
    await loss.selectOption(type);
    if (type === "region_balanced_mse") {
      await expect(
        page.getByLabel("Feature balance: α", { exact: true }),
      ).toHaveValue("0.5");
      await expect(
        page
          .getByText("Advanced region analysis", { exact: true })
          .locator(".."),
      ).not.toHaveAttribute("open", "");
    }
    await expect(page.getByTestId("resolved-training-summary")).toHaveAttribute(
      "aria-busy",
      "false",
    );
    await page
      .locator(".field-loss-controls")
      .screenshot({ path: path.join(evidence, type + ".png") });
    const response = page.waitForResponse((r) =>
      r.url().endsWith("/training/start"),
    );
    await page
      .getByRole("button", { name: "Train model", exact: true })
      .last()
      .click();
    const started = await response;
    expect(started.ok()).toBeTruthy();
    const run = await started.json();
    expect(run.plan.config.field_loss.type).toBe(type);
    await expect
      .poll(
        async () =>
          (
            await (
              await page.request.get(`/api/training/runs/${run.id}`)
            ).json()
          ).state,
        { timeout: 60000 },
      )
      .toBe("completed");
    const final = await (
      await page.request.get(`/api/training/runs/${run.id}`)
    ).json();
    expect(final.plan.config.field_loss.type).toBe(type);
    expect(
      final.plan.field_reconstruction_loss
        .additional_region_queries_per_quantity,
    ).toBe(type === "mse" ? 0 : 256);
    records.push(final);
  }
  fs.writeFileSync(
    path.join(evidence, "actual-runs.json"),
    JSON.stringify(records, null, 2),
  );
});
