// Run after wasm-pack build flare-web --target web.
// Validate the shipped files, including package metadata and WASM initialization.
import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { mkdtemp, readFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';

const source = fileURLToPath(new URL('../../flare-web/', import.meta.url));
const temporary = await mkdtemp(join(tmpdir(), 'flare-wasm-package-'));
try {
  const packed = JSON.parse(execFileSync('npm', [
    'pack', '--json', '--pack-destination', temporary,
  ], { cwd: source, encoding: 'utf8' }));
  assert.equal(packed.length, 1);
  execFileSync('tar', ['-xzf', join(temporary, packed[0].filename), '-C', temporary]);
  const directory = join(temporary, 'package');
  const manifest = JSON.parse(await readFile(join(directory, 'package.json'), 'utf8'));
  const entry = join(directory, manifest.exports['.'].import);
  execFileSync(process.execPath, ['--check', entry], { stdio: 'inherit' });
  // Resolve the public package export from inside its own package scope.
  // No browser/GPU is required when init receives the packaged WASM bytes.
  execFileSync(process.execPath, ['--input-type=module', '--eval', `
    import assert from 'node:assert/strict';
    import { readFile } from 'node:fs/promises';
    const { default: init, FlareEngine } = await import(${JSON.stringify(manifest.name)});
    assert.equal(typeof FlareEngine.load, 'function');
    const wasm = await init({ module_or_path: await readFile('pkg/flare_web_bg.wasm') });
    assert.ok(wasm.memory instanceof WebAssembly.Memory);
    assert.ok(wasm.memory.buffer.byteLength > 0);
    console.log('Packaged ESM syntax, import, and WASM initialization passed');
  `], { cwd: directory, stdio: 'inherit' });
} finally {
  await rm(temporary, { recursive: true, force: true });
}
