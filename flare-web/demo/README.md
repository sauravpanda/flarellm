# Browser SDK demo

```sh
cd flare-web
npm ci
npm run build
python3 -m http.server 8000
```

Open `http://localhost:8000/demo/`, enter a same-origin or CORS-enabled GGUF URL,
and load the model. Optionally provide its original tokenizer JSON. Generation,
loading, and parsing run in the SDK worker; Cancel terminates it immediately.
The next request reloads the configured model. Dispose releases the instance.

See [the SDK guide](../README.md) for model compatibility limits and supported
options. `advanced.html` preserves the previous experimental low-level demo.

The **Use Qwen3-0.6B Q8_0** button fills pinned official model/tokenizer URLs
and selects CPU. See [Qwen3 support](../../QWEN3.md) for the 639 MB download,
512-token browser limit, Apache-2.0 attribution and measured results.
