// Capture dashboard screenshots for the README against the running stack.
//
//   docker compose up -d          # stack must be up, with run 559 seeded
//   node frontend/scripts/screenshot_dashboard.mjs
//
// Override the target with env vars:
//   BASE_URL=http://localhost:5173 RUN_ID=559 node frontend/scripts/screenshot_dashboard.mjs
import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';
import { mkdir } from 'node:fs/promises';
import { chromium } from 'playwright';

const BASE_URL = process.env.BASE_URL ?? 'http://localhost:5173';
const RUN_ID = process.env.RUN_ID ?? '559';
const OUT_DIR = resolve(dirname(fileURLToPath(import.meta.url)), '../../docs/img');

const shots = [
  { name: 'run-list', path: '/', ready: 'table' },
  { name: 'dashboard-winrate', path: `/runs/${RUN_ID}`, tab: 'Win-rate', ready: 'table' },
  { name: 'dashboard-frontier', path: `/runs/${RUN_ID}`, tab: 'Frontier', ready: 'svg .recharts-scatter' },
  { name: 'dashboard-calibration', path: `/runs/${RUN_ID}`, tab: 'Calibration', ready: 'dl' },
];

async function main() {
  await mkdir(OUT_DIR, { recursive: true });
  const browser = await chromium.launch();
  const page = await browser.newPage({ viewport: { width: 1100, height: 620 }, deviceScaleFactor: 2 });

  for (const shot of shots) {
    await page.goto(`${BASE_URL}${shot.path}`, { waitUntil: 'networkidle' });
    if (shot.tab) {
      await page.getByRole('button', { name: shot.tab, exact: true }).click();
    }
    await page.waitForSelector(shot.ready, { timeout: 15_000 });
    await page.waitForTimeout(600); // let charts finish their enter animation
    const file = resolve(OUT_DIR, `${shot.name}.png`);
    await page.screenshot({ path: file }); // viewport-clipped, not full page
    console.log(`wrote ${file}`);
  }

  await browser.close();
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
