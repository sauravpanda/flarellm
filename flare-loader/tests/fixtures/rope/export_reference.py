"""Package output from reference.cpp; expectations never come from Flare.

Usage: python3 export_reference.py /path/to/pinned/llama.cpp /path/to/reference-executable
Optional real-model reference: add /path/to/smollm2.gguf /path/to/tokenizer.json.
"""
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

root = Path(__file__).parent
commit = 'f3f1a8f2760f28325a5ec20c05b171e5b7c83a29'
assert subprocess.check_output(['git', '-C', sys.argv[1], 'rev-parse', 'HEAD'], text=True).strip() == commit
implementation = dict(name='llama.cpp', commit=commit, backend='CPU, Metal disabled, flash attention disabled, f16 KV', context=256)

def run(model, ids):
    with tempfile.TemporaryDirectory() as tmp:
        source, output = Path(tmp) / 'ids.txt', Path(tmp) / 'logits.json'
        source.write_text(' '.join(map(str, ids)))
        subprocess.run([sys.argv[2], str(model), str(source), str(output)], check=True)
        return json.loads(output.read_text())

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

small = root / 'gqa.gguf'
assert digest(small) == '1fc79efebf6e7e74e0ce1a0e218b7621d43ea3e3971c73492f298478b610e78d'
r = run(small, [2,4,7])
r.update(schema=1, referenceImplementation=implementation, modelSha256=digest(small), promptIds=[2,4,7], tolerance=dict(absolute=.04, relative=.002))
(root / 'reference.json').write_text(json.dumps(r, indent=2) + '\n')
if len(sys.argv) > 4:
    import tokenizers
    assert tokenizers.__version__ == '0.22.2'
    model, tokenizer = Path(sys.argv[3]), Path(sys.argv[4])
    assert digest(model) == '48ab3034d0dd401fbc721eb1df3217902fee7dab9078992d66431f09b7750201'
    assert digest(tokenizer) == '9ca9acddb6525a194ec8ac7a87f24fbba7232a9a15ffa1af0c1224fcd888e47c'
    prompt = '<|im_start|>system\nYou are a helpful assistant. Answer briefly.<|im_end|>\n<|im_start|>user\nFacts: The research station is named Aster. Its director is Mira Vale. Who is the director of Aster?<|im_end|>\n<|im_start|>assistant\n'
    ids = tokenizers.Tokenizer.from_file(str(tokenizer)).encode(prompt, add_special_tokens=False).ids
    r = run(model, ids)
    r.update(schema=1, modelSha256=digest(model), modelRevision='593b5a2e04c8f3e4ee880263f93e0bd2901ad47f', modelRepository='HuggingFaceTB/SmolLM2-360M-Instruct-GGUF', modelFile='smollm2-360m-instruct-q8_0.gguf', prompt=prompt, promptIds=ids, eosTokenId=2, tolerance=dict(absolute=.8, relative=.02), referenceImplementation=implementation)
    # Full vocabulary output stays beside the local model for optional browser checks.
    model.with_suffix('.reference.json').write_text(json.dumps(r))
    for step in r['steps']:
        values = step['logits']
        indices = sorted(set(range(0, len(values), 384)) | set(sorted(range(len(values)), key=lambda i: values[i], reverse=True)[:10]))
        step['logitIndices'] = indices
        step['logits'] = [values[i] for i in indices]
    r['vocabSize'] = 49152
    (root / 'smollm2-reference.json').write_text(json.dumps(r, indent=2) + '\n')
