"""Run the independent client on the generated, untrained CI fixture.
Usage: export_fixture_reference.py PINNED_LLAMA_CHECKOUT CLIENT FIXTURE.gguf
"""
import hashlib,json,subprocess,sys,tempfile
from pathlib import Path
root=Path(__file__).parent
reference=json.loads((root/'reference.json').read_text())
assert subprocess.check_output(['git','-C',sys.argv[1],'rev-parse','HEAD'],text=True).strip()==reference['llamaCommit']
with tempfile.TemporaryDirectory() as tmp:
    ids,out=Path(tmp)/'ids.txt',Path(tmp)/'out.json'
    ids.write_text('\n'.join(' '.join(map(str,r['promptIds'])) for r in reference['records']))
    subprocess.run([sys.argv[2],sys.argv[3],str(ids),str(out)],check=True)
    logits=json.loads(out.read_text())
reference['modelSha256']=hashlib.sha256(Path(sys.argv[3]).read_bytes()).hexdigest()
reference['backend']='CPU untrained generated fixture, no GPU, no flash attention, f16 KV'
reference['logitTolerance']=dict(absolute=.04,relative=.002)
reference['scoreAbsoluteTolerance']=.02
for record,values in zip(reference['records'],logits,strict=True):record['logits']=values[:4]
(root/'fixture-reference.json').write_text(json.dumps(reference,indent=2)+'\n')
