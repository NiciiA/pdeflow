import {test,expect} from '@playwright/test';
import path from 'node:path';
test('notebook export imports and reconstructs without reference conditioning',async({page})=>{
 const evidence=path.resolve('../outputs/colab-training-20261027');
 await page.goto('/');await expect(page.getByRole('button',{name:'Play',exact:true})).toBeEnabled();
 await page.getByLabel('Import heat model',{exact:true}).setInputFiles(path.join(evidence,'notebook-run/export/coordinate.pt'));
 await expect(page.getByLabel('Heat model',{exact:true})).toBeVisible();
 await page.getByLabel('Barely trained',{exact:false}).check();await page.getByLabel('Number of measurements').fill('8');
 await page.getByRole('button',{name:'Take measurements',exact:true}).click();await expect(page.getByTestId('measurement-summary')).toContainText('8 measurements');
 const request=page.waitForRequest(r=>r.url().endsWith('/coordinate/reconstruct'));
 await page.getByRole('button',{name:'Run model',exact:true}).click();const submitted=(await request).postDataJSON();
 expect(submitted.rows).toHaveLength(8);expect(submitted.initial).toBeUndefined();expect(submitted.reference).toBeUndefined();expect(submitted.problem.known_fields).toEqual({});
 await expect(page.getByTestId('comparison')).toBeVisible({timeout:60000});await expect(page.getByRole('img',{name:'Model reconstruction',exact:true})).toBeVisible();
 await page.screenshot({path:path.join(evidence,'playground-notebook-import.png'),fullPage:true});
});
