// Portable CI driver. Local interactive testing can call runCI via Chrome CDP.
import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { mkdir, readFile, writeFile, copyFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright';

const here = dirname(fileURLToPath(import.meta.url));
const consumer = resolve(process.argv[2] || '/tmp/flare-consumer');
const output = resolve(process.env.BROWSER_ARTIFACTS || 'browser-artifacts');
const systemGpu = process.argv.includes('--gpu-system');
const gpu = systemGpu || process.argv.includes('--gpu');
// Optional local full-source check; ordinary CI needs only the committed subset.
const originalTokenizer = process.env.ORIGINAL_TOKENIZER_JSON;
const realModel = process.env.REFERENCE_MODEL_GGUF;
const realReference = process.env.REFERENCE_LOGITS_JSON;
const qwen = process.env.REFERENCE_ARCHITECTURE === 'qwen3';
const port = Number(process.env.BROWSER_PORT || 8520);
await mkdir(output, { recursive: true });
let browser, server, context, page;
const logs = [];
const pageErrors = [];
const report = { passed: false, node: process.version, platform: process.platform,
  gpuRequested: gpu, adapterDisabled: process.argv.includes('--disable-adapter'), backend: gpu ? (systemGpu ? 'system WebGPU adapter' : 'SwiftShader software Vulkan (correctness only)') : 'CPU/WASM, GPU disabled' };
try {
  const fixture = JSON.parse(await readFile(resolve(consumer, 'fixture.json'), 'utf8'));
  for (const [name, hash] of Object.entries(fixture.sha256)) {
    assert.equal(createHash('sha256').update(await readFile(resolve(consumer, name))).digest('hex'), hash, `Fixture checksum: ${name}`);
  }
  const reference = JSON.parse(await readFile(resolve(consumer, 'tokenizer-parity/reference.json'), 'utf8'));
  assert.equal(createHash('sha256').update(await readFile(resolve(consumer, 'tokenizer-parity/smollm2-reduced.json'))).digest('hex'), reference.fixtureSha256, 'Tokenizer parity fixture checksum');
  if (originalTokenizer) {
    assert.equal(createHash('sha256').update(await readFile(originalTokenizer)).digest('hex'), qwen ? 'aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4' : reference.originalSha256, 'Original tokenizer checksum');
    await copyFile(originalTokenizer, resolve(consumer, 'original-tokenizer.json'));
  }
  if (realModel) {
    assert(realReference && originalTokenizer, 'Real reference requires model, logits JSON and original tokenizer');
    await copyFile(realModel, resolve(consumer, 'real.gguf'));
    await copyFile(realReference, resolve(consumer, 'real-reference.json'));
  }
  server = spawn('python3', [resolve(here, 'serve.py'), consumer, resolve(consumer, 'fixture.gguf'), '--ci', '--port', String(port)]);
  server.stdout.on('data', data => logs.push(`server: ${data}`));
  server.stderr.on('data', data => logs.push(`server: ${data}`));
  let serverError;
  server.on('error', error => { serverError = error; });
  const url = `http://127.0.0.1:${port}`;
  for (let attempt = 0; ; attempt++) {
    if (serverError) throw serverError;
    assert(server.exitCode === null, `Server exited: ${server.exitCode}`);
    try { if ((await fetch(`${url}/fixture.json`)).ok) break; } catch {}
    assert(attempt < 100, 'Consumer server did not start');
    await new Promise(resolve => setTimeout(resolve, 100));
  }
  browser = await chromium.launch({ headless: true, args: gpu && !process.argv.includes('--disable-adapter')
    ? (systemGpu ? ['--enable-unsafe-webgpu'] : ['--enable-unsafe-webgpu', '--use-angle=swiftshader', '--enable-features=Vulkan', '--use-vulkan=swiftshader'])
    : ['--disable-gpu', '--disable-features=WebGPU'] });
  report.browser = browser.version();
  context = await browser.newContext();
  await context.tracing.start({ screenshots: true, snapshots: true, sources: true });
  page = await context.newPage();
  page.on('console', message => logs.push(`console ${message.type()}: ${message.text()}`));
  page.on('pageerror', error => { pageErrors.push(String(error)); logs.push(`pageerror: ${error.stack}`); });
  page.on('requestfailed', request => logs.push(`requestfailed: ${request.url()} ${request.failure()?.errorText}`));
  await page.goto(url);
  await page.waitForFunction(() => typeof window.runCI === 'function');
  // The timeout also bounds a stuck worker or WASM request.
  await page.evaluate(options => { window.runCI(options); }, { gpu, originalTokenizer: Boolean(originalTokenizer) && !qwen });
  await page.waitForFunction(() => window.ciValidation, null, { timeout: 180000 });
  report.result = await page.evaluate(() => window.ciValidation);
  assert(report.result.passed, JSON.stringify(report.result.error));
  if (realModel) {
    report.realReference = await page.evaluate(({ gpu, qwen }) => new Promise((resolve, reject) => {
      const worker = new Worker('./rope-reference.mjs', { type: 'module' });
      const timer = setTimeout(() => { worker.terminate(); reject(new Error('Real-model reference timed out')); }, 240000);
      worker.onmessage = ({ data }) => { clearTimeout(timer); worker.terminate(); resolve(data); };
      worker.onerror = event => { clearTimeout(timer); worker.terminate(); reject(new Error(event.message)); };
      worker.postMessage({ gpu, real: true, qwen });
    }), { gpu, qwen });
    assert(report.realReference.passed, JSON.stringify(report.realReference));
  }
  if (!gpu) {
    const adapter = await page.evaluate(async () => Boolean(await navigator.gpu?.requestAdapter()));
    assert.equal(adapter, false, 'CPU fallback job unexpectedly has a GPU adapter');
    report.cpuWithoutAdapter = 'passed';
  }
  assert.equal(pageErrors.length, 0, `Uncaught page errors: ${pageErrors.join('; ')}`);
  report.passed = true;
} catch (error) {
  report.error = { message: error.message, stack: error.stack };
  process.exitCode = 1;
} finally {
  if (page) await page.screenshot({ path: resolve(output, 'browser.png'), fullPage: true }).catch(error => logs.push(String(error)));
  if (context) await context.tracing.stop({ path: resolve(output, 'trace.zip') }).catch(error => logs.push(String(error)));
  await browser?.close();
  server?.kill();
  await writeFile(resolve(output, 'result.json'), JSON.stringify(report, null, 2));
  await writeFile(resolve(output, 'browser.log'), logs.join('\n'));
  console.log(JSON.stringify(report, null, 2));
}
