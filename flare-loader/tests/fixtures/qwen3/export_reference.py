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
assert digest(small) == 'f5fc97c646f359091612157e48e54637603e785d884d012fc7e8e0b415309d16'
r = run(small, [2,4,7])
r.update(schema=1, referenceImplementation=implementation, modelSha256=digest(small), promptIds=[2,4,7], tolerance=dict(absolute=.04, relative=.002))
(root / 'reference.json').write_text(json.dumps(r, indent=2) + '\n')

if len(sys.argv) > 4:
    import tokenizers, jinja2
    assert tokenizers.__version__ == '0.22.2'
    model, tokenizer = Path(sys.argv[3]), Path(sys.argv[4])
    assert digest(model) == '9465e63a22add5354d9bb4b99e90117043c7124007664907259bd16d043bb031'
    assert digest(tokenizer) == 'aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4'
    config = tokenizer.with_name('tokenizer_config.json')
    assert digest(config) == 'd5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101'
    template = jinja2.Environment().from_string(json.loads(config.read_text())['chat_template'])
    original = tokenizers.Tokenizer.from_file(str(tokenizer))
    for index, question in enumerate(['What is the capital of France?', 'Reply with exactly the word hello.', 'What is 2 + 2?']):
        messages = [dict(role='user', content=question)]
        prompt = template.render(messages=messages, add_generation_prompt=True, enable_thinking=False)
        ids = original.encode(prompt, add_special_tokens=False).ids
        r = run(model, ids)
        r.update(schema=1, modelSha256=digest(model), modelRevision='23749fefcc72300e3a2ad315e1317431b06b590a', modelRepository='Qwen/Qwen3-0.6B-GGUF', modelFile=model.name, tokenizerSha256=digest(tokenizer), prompt=prompt, messages=messages, promptIds=ids, eosTokenId=151645, tolerance=dict(absolute=.8, relative=.02), referenceImplementation=implementation)
        model.with_suffix(f'.{index}.reference.json').write_text(json.dumps(r))
        print(question, [s['token'] for s in r['steps']], original.decode([s['token'] for s in r['steps']]))
        for step in r['steps']:
            values = step['logits']
            indices = sorted(set(range(0, len(values), 384)) | set(sorted(range(len(values)), key=lambda i: values[i], reverse=True)[:10]))
            step['logitIndices'] = indices
            step['logits'] = [values[i] for i in indices]
        r['vocabSize'] = 151936
        (root / f'real-{index}.json').write_text(json.dumps(r, indent=2) + '\n')
