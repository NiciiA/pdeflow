import { defineConfig } from "@playwright/test";
export default defineConfig({
  testDir: "e2e",
  timeout: 60000,
  workers: 1,
  use: {
    baseURL: "http://127.0.0.1:5173",
    viewport: { width: 1440, height: 1000 },
    trace: "retain-on-failure",
    launchOptions: {
      args: [
        "--disable-gpu",
        "--disable-background-timer-throttling",
        "--disable-renderer-backgrounding",
        "--disable-backgrounding-occluded-windows",
      ],
    },
    reducedMotion: "reduce",
  },
  reporter: "list",
});
