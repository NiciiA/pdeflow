import { test, expect } from "@playwright/test";
import path from "node:path";
test("responsive layout and keyboard playback without a model", async ({
  page,
}) => {
  await page.goto("/");
  await expect(
    page.getByRole("button", { name: "Play", exact: true }),
  ).toBeEnabled();
  await page.screenshot({
    path: path.resolve(
      "../outputs/heat-playground-20261027/main-playground.png",
    ),
    fullPage: true,
  });
  await page.setViewportSize({ width: 390, height: 844 });
  await expect(
    page.getByRole("img", { name: "Heat field", exact: true }),
  ).toBeVisible();
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= window.innerWidth + 1,
    ),
  ).toBeTruthy();
  const play = page.getByRole("button", { name: "Play", exact: true });
  await play.focus();
  await page.keyboard.press("Enter");
  await expect(page.getByTestId("physical-time")).not.toHaveText("t = 0.000");
  await page.getByRole("button", { name: "Pause", exact: true }).click();
  await page.getByRole("button", { name: "Hot spot", exact: true }).click();
  await expect(play).toBeEnabled();
  await page.getByRole("button", { name: "Restart simulation" }).click();
  await expect(page.getByTestId("physical-time")).toHaveText("t = 0.000");
  await page.screenshot({
    path: path.resolve(
      "../outputs/heat-playground-20261027/mobile-playground.png",
    ),
    fullPage: true,
  });
});
