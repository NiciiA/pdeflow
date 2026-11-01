import { test, expect } from "@playwright/test";
test("empty real registry never emits an empty ID or submits a job; draft and technical details survive", async ({
  page,
}) => {
  test.skip(
    process.env.PDEFLOW_EMPTY_REGISTRY !== "1",
    "Run separately against an intentionally empty real API workspace with PDEFLOW_EMPTY_REGISTRY=1",
  );
  await page.goto("/");
  await page.getByRole("button", { name: /^Heat \/ diffusion/ }).click();
  await page.getByRole("button", { name: /2\s*Add information/ }).click();
  await page.getByLabel("Import observations", { exact: true }).setInputFiles({
    name: "manual.csv",
    mimeType: "text/csv",
    buffer: Buffer.from("field,x,y,time,value\nu,0,0,0,0.8\n"),
  });
  await page.waitForTimeout(600);
  await page
    .getByRole("button", { name: "Configure reconstruction", exact: true })
    .click();
  const validation = page.waitForResponse((r) =>
    r.url().endsWith("/api/reconstruction/validate"),
  );
  let jobs = 0;
  page.on("request", (r) => {
    if (r.method() === "POST" && r.url().endsWith("/api/jobs")) jobs++;
  });
  await page
    .getByRole("button", { name: "Validate complete request", exact: true })
    .click();
  const r = await validation;
  expect(r.request().postDataJSON()).not.toHaveProperty("model_id");
  expect(r.request().postDataJSON().model_selection.mode).toBe("auto");
  expect(r.status()).toBe(422);
  await expect(
    page.getByRole("alert").filter({ hasText: "No compatible trained model" }),
  ).toBeVisible();
  await page.getByText("Technical request details", { exact: true }).click();
  await expect(
    page.getByRole("alert").filter({ hasText: "request_id" }),
  ).toBeVisible();
  expect(jobs).toBe(0);
  await expect(
    page.getByRole("button", { name: "Run reconstruction", exact: true }),
  ).toBeDisabled();
  await page
    .getByRole("button", { name: "Clear workspace", exact: true })
    .click();
  await page
    .getByRole("button", { name: "Return without clearing", exact: true })
    .click();
  await page.reload();
  await expect(
    page.getByRole("region", { name: "Full problem summary" }),
  ).toContainText("heat");
  await page.screenshot({
    path: "../outputs/flow-workspace-20261026/empty-registry-fixed.png",
    fullPage: true,
  });
});
