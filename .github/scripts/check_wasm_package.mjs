// Run after the WASM build and npm run build:sdk --prefix flare-web.
// Validate the shipped files, including package metadata and WASM initialization.
import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { mkdtemp, readFile, rm, mkdir, writeFile, cp } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const source = fileURLToPath(new URL('../../flare-web/', import.meta.url));
const retained = process.argv[2] && resolve(process.argv[2]);
const temporary = await mkdtemp(join(tmpdir(), 'flare-wasm-package-'));
try {
  const packed = JSON.parse(execFileSync('npm', [
    'pack', '--json', '--cache', join(temporary, 'cache'), '--pack-destination', temporary,
  ], { cwd: source, encoding: 'utf8' }));
  assert.equal(packed.length, 1);
  const consumer = retained || join(temporary, 'consumer');
  await mkdir(consumer, { recursive: true });
  await writeFile(join(consumer, 'package.json'), JSON.stringify({ private: true, type: 'module' }));
  execFileSync('npm', ['install', '--ignore-scripts', '--no-audit', '--no-fund',
    '--cache', join(temporary, 'cache'), join(temporary, packed[0].filename)], { cwd: consumer, stdio: 'inherit' });
  const directory = join(consumer, 'node_modules/@sauravpanda/flare');
  await cp(join(source, 'tests/browser/index.html'), join(consumer, 'index.html'));
  await cp(join(source, 'tests/browser/consumer.mjs'), join(consumer, 'consumer.mjs'));
  await cp(join(source, 'tests/browser/ci.mjs'), join(consumer, 'ci.mjs'));
  await cp(join(source, 'tests/browser/tokenizer-parity.mjs'), join(consumer, 'tokenizer-parity.mjs'));
  await cp(join(source, '../flare-core/tests/fixtures/tokenizer'), join(consumer, 'tokenizer-parity'), { recursive: true });
  await cp(join(source, 'tests/browser/rope-reference.mjs'), join(consumer, 'rope-reference.mjs'));
  await cp(join(source, '../flare-loader/tests/fixtures/rope'), join(consumer, 'rope-reference'), { recursive: true });
  await cp(join(source, 'tests/browser/fixture.json'), join(consumer, 'fixture.json'));
  await cp(join(source, 'tests/browser/gpu-regression.mjs'), join(consumer, 'gpu-regression.mjs'));
  await cp(join(source, 'tests/browser/probe-worker.mjs'), join(consumer, 'probe-worker.mjs'));
  await cp(join(source, 'tests/browser/consumer-typecheck.mts'), join(consumer, 'consumer-typecheck.mts'));
  execFileSync(process.execPath, [join(source, 'node_modules/typescript/bin/tsc'),
    '--noEmit', '--strict', '--target', 'ES2022', '--module', 'NodeNext',
    '--lib', 'ES2022,DOM,ESNext.Disposable', 'consumer-typecheck.mts'],
    { cwd: consumer, stdio: 'inherit' });
  const manifest = JSON.parse(await readFile(join(directory, 'package.json'), 'utf8'));
  const entry = join(directory, manifest.exports['.'].import);
  execFileSync(process.execPath, ['--check', entry], { stdio: 'inherit' });
  execFileSync(process.execPath, ['--check', join(directory, manifest.exports['./worker'].import)], { stdio: 'inherit' });
  // Resolve the public exports from a separate installed consumer.
  // No browser/GPU is required when init receives the packaged WASM bytes.
  execFileSync(process.execPath, ['--input-type=module', '--eval', `
    import assert from 'node:assert/strict';
    import { readFile } from 'node:fs/promises';
    const { default: init, FlareEngine, Flare } = await import(${JSON.stringify(manifest.name)});
    assert.equal(typeof FlareEngine.load, 'function');
    assert.equal(typeof Flare.init, 'function');
    const low = await import('@sauravpanda/flare/wasm');
    assert.equal(low.FlareEngine, FlareEngine);
    assert.ok(import.meta.resolve('@sauravpanda/flare/worker').endsWith('/dist/worker.js'));
    const wasm = await init({ module_or_path: await readFile('node_modules/@sauravpanda/flare/pkg/flare_web_bg.wasm') });
    assert.ok(wasm.memory instanceof WebAssembly.Memory);
    assert.ok(wasm.memory.buffer.byteLength > 0);
    console.log('Packaged SDK types, ESM exports, worker syntax, and WASM initialization passed');
  `], { cwd: consumer, stdio: 'inherit' });
  if (retained) console.log(`Browser consumer installed at ${retained}`);
} finally {
  await rm(temporary, { recursive: true, force: true });
}
