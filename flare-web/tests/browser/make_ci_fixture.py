"""Generate the existing numerical fixture and a deliberately synthetic tokenizer."""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

root = Path(__file__).parent
out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
subprocess.run([sys.executable, str(root / 'make_gpu_fixture.py'), str(out / 'fixture.gguf')], check=True)
vocab = {f'<t{i}>': i for i in range(128)}
for token, index in {'H': 2, 'e': 4, 'l': 7, 'o': 9, 'â': 36, 'Ĥ': 28, '¬': 20}.items():
    del vocab[f'<t{index}>']
    vocab[token] = index
(out / 'tokenizer.json').write_text(json.dumps({'model': {'vocab': vocab, 'merges': []}}, ensure_ascii=True) + '\n')
manifest = json.loads((root / 'fixture.json').read_text())
for name, expected in manifest['sha256'].items():
    actual = hashlib.sha256((out / name).read_bytes()).hexdigest()
    if actual != expected:
        raise SystemExit(f'{name}: SHA-256 {actual} != pinned {expected}')
print('Fixture checksums verified')

# Same Qwen3 architecture fixture, with a generated full-size vocabulary so
# the original tokenizer/template can exercise actual SDK decisions in CI.
subprocess.run([sys.executable, str(root.parents[2] / 'flare-loader/tests/fixtures/qwen3/generate.py'), '--decision', str(out / 'decision-fixture.gguf')], check=True)
decision_reference = json.loads((out / 'decision/fixture-reference.json').read_text())
assert hashlib.sha256((out / 'decision-fixture.gguf').read_bytes()).hexdigest() == decision_reference['modelSha256']
