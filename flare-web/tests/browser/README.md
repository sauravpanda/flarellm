# Installed package browser check

Build WASM and the SDK first. Install the actual tarball into a retained consumer:

```sh
node .github/scripts/check_wasm_package.mjs /tmp/flare-consumer
cp /absolute/path/tokenizer.json /tmp/flare-consumer/tokenizer.json
python3 flare-web/tests/browser/serve.py /tmp/flare-consumer /absolute/path/model.gguf
```

Open `http://127.0.0.1:8520` in an existing Chrome instance and call `await run()`
from its console (or through an authorized CDP harness). Results appear on the
page and in `window.validation`. Call `run('webgpu')` separately to exercise
async GPU decode. A running UI tick counter checks main-thread responsiveness.
The model stays local. No automated model download or browser launch is involved.

Coverage: failed initialization and load, download progress, actual inference
chunks/result equality, cancellation after a decoded token, a follow-up request,
reset, cancellation while a long prompt prefills, BUSY rejection, slow download
abort, reload, disposal, and Unicode bytes from the installed WASM binding.
The fixture defaults to `/tokenizer.json`; pass a second argument
`null` to `run('cpu', null)` for a supported embedded SentencePiece vocabulary.
Output quality is deliberately reported separately from lifecycle assertions;
these checks do not establish parity with a reference engine.
