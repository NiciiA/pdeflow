import { experts } from "./disclosure";
import { test, expect } from "@playwright/test";
import fs from "node:fs/promises";
for (const dimension of [2, 3])
  test(`chemical ${dimension}D automatic routing, real one-species reconstruction and reaction-law rejection`, async ({
    page,
  }) => {
    const errors: string[] = [];
    let submissions = 0;
    page.on("pageerror", (e) => errors.push(e.message));
    page.on("request", (r) => {
      if (r.url().endsWith("/api/jobs") && r.method() === "POST") submissions++;
    });
    await page.goto("/");
    await page
      .getByRole("button", { name: "Reset draft", exact: true })
      .click();
    await experts(page);
    await experts(page);
    await page.getByRole("button", { name: /^A\+B/ }).click();
    await experts(page);
    if (dimension === 3)
      await page
        .getByLabel("Spatial dimension", { exact: true })
        .selectOption("3");
    await page.getByLabel("Value k_f", { exact: true }).fill(".8");
    await expect(page.getByText("checkpoint: yes")).toBeVisible();
    await expect(page.getByLabel("Routing report")).toContainText(
      "selected specialist: chemistry",
    );
    await expect(page.getByText(/Checkpoint data mode: online/)).toBeVisible();
    const complete = JSON.parse(
      await page.getByLabel("Complete mathematical request JSON").inputValue(),
    );
    const exported = page.waitForEvent("download");
    await page.getByRole("button", { name: "Export complete request" }).click();
    await experts(page);
    const requestFile = await (await exported).path();
    await page
      .getByLabel("Import complete request")
      .setInputFiles(requestFile!);
    expect(
      JSON.parse(
        await page
          .getByLabel("Complete mathematical request JSON")
          .inputValue(),
      ),
    ).toEqual(complete);
    await expect(page.getByText("checkpoint: yes")).toBeVisible();
    await page.getByLabel("Exported training mode").selectOption("hybrid");
    const configDownload = page.waitForEvent("download");
    await page
      .getByRole("button", { name: "Export training configuration" })
      .click();
    await experts(page);
    const config = JSON.parse(
      await fs.readFile((await (await configDownload).path())!, "utf8"),
    );
    expect(config.provider.mode).toBe("hybrid");
    expect(config.provider.distribution_version).toBe("physical_v1");
    expect(config.provider.solver_rtol).toBe(1e-10);
    expect(config.specialist).toBe("chemistry");
    const commandsDownload = page.waitForEvent("download");
    await page
      .getByRole("button", { name: "Export training commands" })
      .click();
    await experts(page);
    const commands = await fs.readFile(
      (await (await commandsDownload).path())!,
      "utf8",
    );
    expect(commands).toContain("prepare-evaluation");
    expect(commands).toContain("--evaluation");
    expect(commands).not.toContain(" --data ");
    expect(submissions).toBe(0);
    await page
      .getByRole("button", { name: "Continue to observations" })
      .click();
    await experts(page);
    await page
      .getByRole("button", { name: "Generate synthetic observations" })
      .click();
    await experts(page);
    await expect(
      page.getByText("120 measurements", { exact: true }),
    ).toBeVisible();
    const csvDownload = page.waitForEvent("download");
    await page.getByRole("button", { name: "Export CSV" }).click();
    await experts(page);
    const csv = await fs.readFile((await (await csvDownload).path())!, "utf8");
    const onlyA = csv
      .split("\n")
      .filter((line, i) => i === 0 || line.startsWith("a,"))
      .join("\n");
    await page.getByLabel("Import observations").setInputFiles({
      name: "only-a.csv",
      mimeType: "text/csv",
      buffer: Buffer.from(onlyA),
    });
    await expect(
      page.getByText("40 measurements", { exact: true }),
    ).toBeVisible();
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
      page.getByText("No evaluation reference attached", { exact: true }),
    ).toBeVisible();
    await expect(
      page.getByRole("table", {
        name: "Per-field reconstruction errors",
        exact: true,
      }),
    ).toHaveCount(0);
    await page.getByLabel("Physical field", { exact: true }).selectOption("2");
    await page.getByLabel("Physical time selection").fill("2");
    if (dimension === 3) {
      await page.getByLabel("Slice normal").selectOption("0");
      await page.getByLabel("Slice index").fill("3");
      await expect(
        page.getByRole("img", { name: /Reconstruction, y \/ z/ }),
      ).toBeVisible();
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
    await expect(
      page.getByRole("img", { name: /Absolute error/ }),
    ).toBeVisible();
    await page.screenshot({
      path: `../outputs/workbench-redesign-20261025/browser/regression/browser-chemistry-${dimension}d.png`,
      fullPage: true,
    });
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
    await page.getByRole("button", { name: /1\s*Define problem/ }).click();
    await page
      .getByRole("heading", {
        name: "Define the physical problem",
        exact: true,
      })
      .waitFor({ state: "visible" });
    await experts(page);
    await page.getByRole("button", { name: "Show LaTeX source" }).click();
    await experts(page);
    await page.getByLabel("LaTeX source 4").fill("r=k_f a b-k_r c+a");
    await expect(page.getByText(/No implemented specialist/)).toBeVisible();
    expect(
      JSON.parse(
        await page
          .getByLabel("Complete mathematical request JSON")
          .inputValue(),
      ).equations[3],
    ).toContain("+a");
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
    expect(errors).toEqual([]);
  });
