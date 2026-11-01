import { experts } from "./disclosure";
import { test, expect } from "@playwright/test";
test("real locally trained checkpoint is discovered, refreshed, reconstructed and reopened", async ({
  page,
}) => {
  await page.goto("/");
  await page.getByRole("button", { name: "Reset draft", exact: true }).click();
  await experts(page);
  await experts(page);
  const response = await page.request.get("/api/models");
  expect(response.ok()).toBeTruthy();
  const rows = await response.json();
  const model = rows.find(
    (m: { run_name: string; export_kinds: string[] }) =>
      m.run_name === "workflow-journey" && m.export_kinds.includes("latest"),
  );
  expect(model).toBeTruthy();
  expect(model.training.updates).toBe(20);
  await page.getByRole("button", { name: /^Heat \/ diffusion/ }).click();
  await experts(page);
  await page.getByLabel("Registered checkpoint").selectOption(model.id);
  await expect(page.getByText("checkpoint: yes")).toBeVisible();
  await page.getByLabel("Diffusivity", { exact: true }).fill("0.08");
  await page.getByRole("button", { name: "Refresh checkpoints" }).click();
  await experts(page);
  await expect(page.getByLabel("Diffusivity", { exact: true })).toHaveValue(
    "0.08",
  );
  await expect(page.getByText("checkpoint: yes")).toBeVisible();
  await page.getByRole("button", { name: "Continue to observations" }).click();
  await experts(page);
  await page
    .getByRole("button", { name: "Generate synthetic observations" })
    .click();
  await experts(page);
  await page.getByRole("button", { name: "Configure reconstruction" }).click();
  await experts(page);
  await page.getByLabel("Generation steps", { exact: true }).fill("4");
  await page.getByLabel("Sample count", { exact: true }).fill("2");
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
    page.getByText("No evaluation reference attached", { exact: true }),
  ).toBeVisible();
  await expect(
    page.getByRole("button", {
      name: "Attach synthetic reference for evaluation",
    }),
  ).toBeVisible();
  await page.screenshot({
    path: "../outputs/workbench-redesign-20261025/browser/managed-result.png",
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
  await expect(
    page.getByRole("heading", { name: "Inspect the reconstruction" }),
  ).toBeVisible();
  await expect(
    page.getByText("No evaluation reference attached", { exact: true }),
  ).toBeVisible();
});
