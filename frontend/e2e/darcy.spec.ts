import { experts } from "./disclosure";
import { test, expect } from "@playwright/test";

async function enter(
  page: import("@playwright/test").Page,
  family = "darcy",
  dim = 2,
) {
  await page.goto("/");
  await page.getByRole("button", { name: "Reset draft", exact: true }).click();
  await experts(page);
  await experts(page);
  await page
    .getByRole("button", {
      name: family === "darcy" ? /^Darcy/ : /^Heat \/ diffusion/,
    })
    .click();
  await experts(page);
  if (dim === 3) {
    await page
      .getByLabel("Spatial dimension", { exact: true })
      .selectOption("3");
  }
  if (family === "darcy")
    await page
      .getByRole("button", { name: "Set known mean-zero sinusoidal forcing" })
      .click();
  await experts(page);
  await expect(page.getByText("checkpoint: yes")).toBeVisible();
  await expect(page.getByLabel("Registered checkpoint")).toHaveValue("");
  await expect(page.getByLabel("Routing report")).toContainText(
    `selected specialist: ${family}`,
  );
}
for (const dim of [2, 3])
  test(`real ${dim}D Darcy auto-routing, joint fields, detached truth and static slices`, async ({
    page,
  }) => {
    const errors: string[] = [];
    page.on("pageerror", (e) => errors.push(e.message));
    await enter(page, "darcy", dim);
    await page.getByRole("button", { name: "Show LaTeX source" }).click();
    await experts(page);
    await page.getByLabel("LaTeX source 1").fill("-div(a grad(u))-f=0");
    await expect(page.getByText("checkpoint: yes")).toBeVisible();
    await page
      .getByRole("button", { name: "Continue to observations" })
      .click();
    await experts(page);
    await page
      .getByRole("button", { name: "Generate synthetic observations" })
      .click();
    await experts(page);
    await expect(
      page.getByText("20 measurements", { exact: true }),
    ).toBeVisible();
    const download = page.waitForEvent("download");
    await page.getByRole("button", { name: "Export CSV" }).click();
    await experts(page);
    await page
      .getByLabel("Import observations")
      .setInputFiles((await (await download).path())!);
    await page
      .getByRole("button", { name: "Configure reconstruction" })
      .click();
    await experts(page);
    await page.getByLabel("Generation steps", { exact: true }).fill("3");
    await page
      .getByRole("button", { name: "Validate complete request" })
      .click();
    await experts(page);
    await expect(page.getByText("Compatible request")).toBeVisible();
    await page
      .getByRole("button", { name: "Run reconstruction", exact: true })
      .click();
    await experts(page);
    await expect(
      page.getByText("Physical diagnostics", { exact: true }),
    ).toBeVisible({ timeout: 30000 });
    await expect(
      page.getByText("Static fields · no physical time"),
    ).toBeVisible();
    await expect(page.getByLabel("Physical time selection")).toHaveCount(0);
    await page.getByLabel("Physical field", { exact: true }).selectOption("1");
    await expect(
      page.getByRole("img", { name: /Reconstruction, .*static field/ }),
    ).toBeVisible();
    if (dim === 3) {
      await page.getByLabel("Slice normal").selectOption("0");
      await page.getByLabel("Slice index").fill("3");
      await expect(
        page.getByRole("img", { name: /Reconstruction, y \/ z, static field/ }),
      ).toBeVisible();
    }
    await page
      .getByRole("button", {
        name: "Attach synthetic reference for evaluation",
      })
      .click();
    await experts(page);
    await expect(
      page.getByRole("img", { name: /Absolute error/ }),
    ).toBeVisible();
    await page.getByLabel("Physical field", { exact: true }).selectOption("0");
    await page.screenshot({
      path: `../outputs/workbench-redesign-20261025/browser/regression/browser-darcy-${dim}d.png`,
      fullPage: true,
    });
    await page
      .getByRole("button", { name: "Remove evaluation reference" })
      .click();
    await experts(page);
    await expect(
      page.getByText("No evaluation reference attached", { exact: true }),
    ).toBeVisible();
    expect(errors).toEqual([]);
  });

test("edited family, wrong override, dimensions/boundaries and unsupported terms revalidate", async ({
  page,
}) => {
  await enter(page, "heat");
  const models = await (await page.request.get("/api/models")).json();
  const heatOption = models.find(
    (m: { specialist: string; problem: { grid: { shape: number[] } } }) =>
      m.specialist === "heat" && m.problem.grid.shape.length === 2,
  );
  expect(heatOption).toBeTruthy();
  await page.getByRole("button", { name: /^Darcy/ }).click();
  await experts(page);
  await page.getByRole("button", { name: "Apply reviewed changes" }).click();
  await experts(page);
  await page
    .getByRole("button", { name: "Set known mean-zero sinusoidal forcing" })
    .click();
  await experts(page);
  await expect(page.getByLabel("Routing report")).toContainText(
    "selected specialist: darcy",
  );
  await expect(page.getByText("checkpoint: yes")).toBeVisible();
  await page.getByLabel("Registered checkpoint").selectOption(heatOption.id);
  await expect(
    page.getByText(/Explicit checkpoint override is incompatible/),
  ).toBeVisible();
  await expect(page.getByText("checkpoint: unavailable")).toBeVisible();
  await page.getByLabel("Registered checkpoint").selectOption("");
  await expect(page.getByText("checkpoint: yes")).toBeVisible();
  await page.getByLabel("Spatial dimension", { exact: true }).selectOption("1");
  await expect(page.getByText("checkpoint: unavailable")).toBeVisible();
  await page.getByLabel("Spatial dimension", { exact: true }).selectOption("2");
  await page
    .getByRole("button", { name: "Set known mean-zero sinusoidal forcing" })
    .click();
  await experts(page);
  await page.getByLabel("Boundary convention").selectOption("neumann");
  await expect(
    page.getByText(/Implemented specialists require known periodic/),
  ).toBeVisible();
  await page.getByLabel("Boundary convention").selectOption("periodic");
  await expect(page.getByText("checkpoint: yes")).toBeVisible();
  await page.getByRole("button", { name: "Show LaTeX source" }).click();
  await experts(page);
  await page.getByLabel("LaTeX source 1").fill("-div(a grad(u))+u=f");
  await expect(page.getByText(/No implemented specialist/)).toBeVisible();
  expect(
    JSON.parse(
      await page.getByLabel("Complete mathematical request JSON").inputValue(),
    ).equations[0],
  ).toContain("+u");
  await page.getByRole("button", { name: /3\s*Reconstruct/ }).click();
  if (
    await page
      .getByRole("button", {
        name: "Adjust reconstruction settings",
        exact: true,
      })
      .isVisible()
  )
    await page
      .getByRole("button", {
        name: "Adjust reconstruction settings",
        exact: true,
      })
      .click();
  await experts(page);
  await expect(
    page.getByRole("button", { name: "Run reconstruction", exact: true }),
  ).toBeDisabled();
});
