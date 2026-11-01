import { test, expect as baseExpect } from "@playwright/test";
import type { Page } from "@playwright/test";
// Historical discovery includes 85 real exports on this workstation. Initial
// requests can take several seconds; wait for readiness rather than fixed sleeps.
const expect = baseExpect.configure({timeout:20000});
const evidence = "../outputs/integrated-training-20261027/browser";

async function openTraining(page: Page) {
  await page.goto("/");
  await page.getByRole("button", { name: "Models & Training", exact: true }).click();
  await expect(page.getByRole("heading", { name: "Train model", exact: true })).toBeVisible();
  await page.getByLabel("Training device", { exact: true }).selectOption("cpu");
}

async function train(page: Page, model: string) {
  await page.getByLabel("Training model", { exact: true }).selectOption(model);
  await expect(page.getByRole("button", { name: "Train model", exact: true })).toBeEnabled();
  const response = page.waitForResponse(r => r.url().endsWith("/api/training/start") && r.request().method() === "POST");
  await page.getByRole("button", { name: "Train model", exact: true }).click();
  const started = await response;
  expect(started.status()).toBe(202);
  expect(started.request().postDataJSON()).toMatchObject({ model, budget:"pipeline-check", device:"cpu" });
  const id = (await started.json()).id as string;
  await expect(page.getByLabel("Integrated training run",{exact:true})).toHaveValue(id);
  await expect.poll(async()=>(await(await page.request.get(`/api/training/runs/${id}`)).json()).state,{timeout:60000}).toBe("completed");
  await expect(page.getByRole("button", { name:"Use this model", exact:true })).toBeVisible({timeout:60000});
  const result = await page.request.get(`/api/training/runs/${id}`);
  expect(result.ok()).toBeTruthy();
  const row = await result.json();
  expect(row.state).toBe("completed");
  expect(row.plan.resolved_device).toBe("cpu");
  expect(row.validation.split).toBe("validation");
  expect(row.plan.pipeline_check).toBe(true);
  await expect(page.getByRole("img", { name:/Absolute error/ }).first()).toBeVisible();
  return row;
}

test("real one-action resolution-flexible training, reload, explicit use and numerical reconstruction", async ({page}) => {
  test.setTimeout(120000);
  const errors:string[]=[];
  page.on("pageerror", e=>errors.push(e.message));
  let launches=0;page.on("request", r=>{if(r.url().endsWith("/api/training/start")) launches++;});
  await openTraining(page);
  await expect(page.getByText(/unsuitable for judging accuracy/).first()).toBeVisible();
  await page.getByRole("button",{name:"Train model",exact:true}).scrollIntoViewIfNeeded();
  await page.screenshot({path:`${evidence}/training-form.png`});
  const row = await train(page,"resolution-flexible");
  expect(row.finished_stages).toContain("representation");
  expect(row.finished_stages).toContain("flow");
  expect(row.representation_quality.validation_mse).toBeGreaterThan(0);
  await page.screenshot({path:`${evidence}/resolution-flexible-result.png`,fullPage:true});
  await page.reload();
  await expect(page.getByRole("button",{name:"Use this model",exact:true})).toBeVisible({timeout:20000});
  expect(launches).toBe(1);
  await page.getByRole("button",{name:"Use this model",exact:true}).click();
  await expect(page.getByLabel("Coordinate checkpoint",{exact:true})).toHaveValue(row.model_id);
  await page.getByLabel("Coordinate flow steps",{exact:true}).fill("2");
  const button=page.getByRole("button",{name:"Reconstruct with coordinate specialist",exact:true});
  await expect(button).toBeEnabled();
  const response=page.waitForResponse(r=>r.url().endsWith("/api/coordinate/reconstruct")&&r.request().method()==="POST");
  await button.click();const launched=await response;expect(launched.status()).toBe(202);
  const id=(await launched.json()).id;
  await expect.poll(async()=>(await (await page.request.get(`/api/jobs/${id}`)).json()).status,{timeout:30000}).toBe("succeeded");
  const result=await(await page.request.get(`/api/coordinate/jobs/${id}/result`)).json();
  expect(result.metrics.no_reference_input).toBe(true);
  expect(result.metrics.checkpoint_updates).toBe(4);
  await page.screenshot({path:`${evidence}/resolution-flexible-use.png`,fullPage:true});
  expect(errors).toEqual([]);
});

test("real fixed-grid training, validation preview, use model and retained numerical inverse", async({page})=>{
  test.setTimeout(120000);
  const errors:string[]=[];page.on("pageerror",e=>errors.push(e.message));
  await openTraining(page);
  const row=await train(page,"fixed-grid");
  expect(row.finished_stages).not.toContain("representation");
  expect(row.plan.updates.fixed).toBe(2);
  await page.screenshot({path:`${evidence}/fixed-grid-result.png`,fullPage:true});
  await page.getByRole("button",{name:"Use this model",exact:true}).click();
  await page.getByRole("button",{name:/2\s*Add information/}).click();
  await page.getByRole("button",{name:"Load visible reference defaults",exact:true}).click();
  await page.getByRole("button",{name:"Solve reference",exact:true}).click();
  await expect(page.getByRole("button",{name:"Generate observations",exact:true})).toBeEnabled();
  await page.getByRole("button",{name:"Generate observations",exact:true}).click();
  await expect(page.getByRole("button",{name:"Reconstruct from measurements",exact:true})).toBeEnabled();
  await page.getByRole("button",{name:"Reconstruct from measurements",exact:true}).click();
  await page.getByRole("button",{name:"Validate complete request",exact:true}).click();
  await expect(page.getByRole("button",{name:"Run reconstruction",exact:true})).toBeEnabled();
  const response=page.waitForResponse(r=>r.url().endsWith("/api/jobs")&&r.request().method()==="POST");
  await page.getByRole("button",{name:"Run reconstruction",exact:true}).click();
  const launched=await response;expect(launched.status()).toBe(202);
  expect(launched.request().postDataJSON().model_selection.id).toBe(row.model_id);
  const id=(await launched.json()).id;
  await expect.poll(async()=>(await(await page.request.get(`/api/jobs/${id}`)).json()).status,{timeout:30000}).toBe("succeeded");
  await page.screenshot({path:`${evidence}/fixed-grid-use.png`,fullPage:true});
  expect(errors).toEqual([]);
});

test("rendered safe stop and resume keeps one persisted training run",async({page})=>{
  test.setTimeout(120000);
  await openTraining(page);
  await page.getByLabel("Training model",{exact:true}).selectOption("resolution-flexible");
  await page.getByText("Advanced settings and architecture",{exact:true}).click();
  await page.getByLabel("Representation updates",{exact:true}).fill("80");
  await expect(page.getByRole("button",{name:"Train model",exact:true})).toBeEnabled();
  const started=page.waitForResponse(r=>r.url().endsWith("/api/training/start")&&r.request().method()==="POST");
  await page.getByRole("button",{name:"Train model",exact:true}).click();
  const id=(await(await started).json()).id;
  const stop=page.getByRole("button",{name:"Stop safely",exact:true});
  await expect(stop).toBeVisible();await stop.click();
  const resume=page.getByRole("button",{name:"Resume training",exact:true});
  await expect(resume).toBeVisible({timeout:30000});
  await page.screenshot({path:`${evidence}/stopped.png`,fullPage:true});
  await page.reload();await expect(resume).toBeVisible({timeout:20000});await resume.click();
  await expect(page.getByRole("button",{name:"Use this model",exact:true})).toBeVisible({timeout:60000});
  const row=await(await page.request.get(`/api/training/runs/${id}`)).json();
  expect(row.state).toBe("completed");expect(row.attempts).toBe(2);
});
