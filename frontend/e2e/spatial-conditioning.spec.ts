import {test,expect} from '@playwright/test';
import fs from 'node:fs';
import path from 'node:path';
test.use({baseURL:'http://127.0.0.1:5199'});
test('rendered B0–B3 selectors reach real resolved training configurations',async({page})=>{
 const out=path.resolve('../outputs/spatial-conditioning-20261101/browser');fs.mkdirSync(out,{recursive:true});const evidence:unknown[]=[];
 await page.goto('/');await page.getByRole('button',{name:'Train model',exact:true}).first().click();
 await expect(page.getByLabel('Conditional experiment (B)')).toBeVisible();
 for(const [value,version,beta] of [['spatial-aligned','spatial_v1',.01],['pooled-aligned','pooled_v1',.01],['spatial-fm','spatial_v1',0],['pooled-fm','pooled_v1',0]] as const){
  const response=page.waitForResponse(r=>r.url().endsWith('/training/plan')&&r.request().method()==='POST'&&JSON.parse(r.request().postData()||'{}').advanced?.joint?.conditioning?.version===version&&((JSON.parse(r.request().postData()||'{}').advanced?.joint?.observation_auxiliary?.beta)||0)===beta);
  await page.getByLabel('Conditional experiment (B)').selectOption(value);const r=await response;expect(r.ok()).toBeTruthy();const payload=r.request().postDataJSON();const resolved=await r.json();
  expect(resolved.config.joint.conditioning.version).toBe(version);expect(resolved.config.joint.observation_auxiliary.beta||0).toBe(beta);expect(resolved.config.field_loss.type).toBe('mse');expect(resolved.config.family_weights).toEqual({heat:.5,wave:.5});evidence.push({value,payload,resolved});
 }
 const finalResponse=page.waitForResponse(r=>r.url().endsWith('/training/plan')&&r.request().method()==='POST'&&JSON.parse(r.request().postData()||'{}').advanced?.joint?.conditioning?.version==='spatial_v1');
 await page.getByLabel('Conditional experiment (B)').selectOption('spatial-aligned');await finalResponse;await expect(page.getByTestId('resolved-training-summary')).toHaveAttribute('aria-busy','false');await expect(page.getByLabel('Conditional experiment (B)')).toHaveValue('spatial-aligned');
 await page.screenshot({path:path.join(out,'joint-spatial-summary.png'),fullPage:true});fs.writeFileSync(path.join(out,'resolved-plans.json'),JSON.stringify(evidence,null,2));
 expect(page.getByText('B3 · Spatial + observation-aligned')).toBeTruthy();
});
