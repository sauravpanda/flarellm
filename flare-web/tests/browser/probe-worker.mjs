import init, { cache_model, load_cached_model, delete_cached_model, storage_estimate,
  device_info, supports_webnn, supports_webtransport, FlareProgressiveLoader } from './node_modules/@sauravpanda/flare/pkg/flare_web.js';
try {
  await init();
  const key = `sdk-test-${crypto.randomUUID()}`;
  let bytes;
  try {
    await cache_model(key, new Uint8Array([1, 2, 3]));
    bytes = Array.from(await load_cached_model(key));
  } finally { await delete_cached_model(key); }
  const loader = new FlareProgressiveLoader('/missing.gguf');
  let fetchError;
  try { await loader.load(() => {}); } catch (error) { fetchError = error.message; }
  finally { loader.free(); }
  postMessage({ bytes, storage: JSON.parse(await storage_estimate()), device: JSON.parse(device_info()),
    webnn: supports_webnn(), webtransport: supports_webtransport(), fetchError });
} catch (error) { postMessage({ error: String(error) }); }
