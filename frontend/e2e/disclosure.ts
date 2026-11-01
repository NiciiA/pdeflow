import type { Page } from "@playwright/test";
// Legacy expert workflows deliberately inspect the complete configuration.
// Primary journeys in design.spec.ts keep progressive disclosure closed.
export async function experts(page: Page) {
  if (await page.getByRole("dialog").isVisible()) return;
  for (let pass = 0; pass < 2; pass++) {
    for (const detail of await page.locator("details").all()) {
      const summary = detail.locator(":scope > summary");
      if (
        (await summary.isVisible()) &&
        (await detail.getAttribute("open")) === null
      )
        await summary.click();
    }
  }
}
export async function physical(page: Page) {
  const summary = page.getByText(
    "Physical verification, process status & actions",
    { exact: true },
  );
  await summary.waitFor({ state: "visible" });
  const detail = page.locator("details").filter({ has: summary });
  if ((await detail.getAttribute("open")) === null) await summary.click();
}
