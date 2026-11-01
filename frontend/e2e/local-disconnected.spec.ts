import { test, expect } from "@playwright/test";
test("real frontend explains an absent backend", async ({ page }) => {
  await page.goto("/");
  const notice = page.getByRole("alert", { name: "Backend connection" });
  await expect(notice).toContainText("Backend unavailable.");
  await expect(notice).toContainText("http://127.0.0.1:8000");
  await expect(notice).toContainText("pdeflow serve");
  await expect(
    page.getByRole("button", { name: "Refresh checkpoints" }),
  ).toBeVisible();
  await page.screenshot({
    path: "../outputs/local-workflow-20261025/browser/disconnected.png",
    fullPage: true,
  });
});
