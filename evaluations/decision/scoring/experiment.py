#!/usr/bin/env python3
"""Opt-in Q8 prompt experiment; normal CI never downloads or loads weights."""
import argparse
import json
import math
import platform
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT.parent/'comparison'))
from common import attempts, digest, load_dataset, read, summarize, validate_result, write
from prompts import VARIANTS, prepare

LLAMA = 'f3f1a8f2760f28325a5ec20c05b171e5b7c83a29'
MODEL = '9465e63a22add5354d9bb4b99e90117043c7124007664907259bd16d043bb031'
TOKENIZER = 'aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4'
CONFIG = 'd5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101'


def score(logits, choices):
    if len(logits) != len(choices) or not all(math.isfinite(z) for z in logits):
        raise ValueError('invalid candidate logits')
    values = [math.exp(z-max(logits)) for z in logits]
    probabilities = [v/math.fsum(values) for v in values]
    best = max(range(len(logits)),key=logits.__getitem__)
    return dict(choice=choices[best],scores=[dict(choice=c,logit=z,score=p)
                for c,z,p in zip(choices,logits,probabilities,strict=True)])


def describe(data, run, sha):
    result = summarize(data,run,sha)
    rows = [r for r in run['records'] if 'result' in r]
    result['labelMass'] = dict(completed=len(rows),mean=statistics.mean(r['labelMass'] for r in rows) if rows else None,
                              minimum=min((r['labelMass'] for r in rows),default=None),
                              topTokenOutsideLabels=sum(r['top'][0]['tokenId'] not in r['labelIds'] for r in rows))
    return result


def selection(data, runs, sha):
    if set(runs) != set(VARIANTS):
        raise ValueError('selection requires exactly the four frozen candidates')
    summaries = {v:describe(data,runs[v],sha) for v in VARIANTS}
    def key(v):
        s=summaries[v]
        return (min(x['accuracy'] for x in s['byOrder'].values()),
                s['pooledPaired']['accuracy'],-s['pooledPaired']['brier'],-VARIANTS.index(v))
    return max(VARIANTS,key=key),summaries


def run(args):
    import jinja2
    import tokenizers
    if tokenizers.__version__!='0.22.2' or jinja2.__version__!='3.1.6':
        raise ValueError('use pinned tokenizers/Jinja versions')
    if digest(args.model)!=MODEL or digest(args.tokenizer)!=TOKENIZER or digest(args.tokenizer.with_name('tokenizer_config.json'))!=CONFIG:
        raise ValueError('model/tokenizer hash mismatch')
    commit=subprocess.check_output(['git','-C',str(args.llama),'rev-parse','HEAD'],text=True).strip()
    if commit!=LLAMA:
        raise ValueError('wrong llama.cpp revision')
    data=load_dataset(args.dataset)
    frozen=None
    if args.selection:
        frozen=read(args.selection)
        if frozen['promptsSha256']!=digest(ROOT/'prompts.py'):
            raise ValueError('prompt changed after selection')
        variants=list(dict.fromkeys(['baseline',frozen['winner']]))
    else:
        if args.dataset.resolve()!=(ROOT.parent/'comparison/dev.json').resolve():
            raise ValueError('candidate sweep is restricted to development data')
        variants=VARIANTS
    tokenizer=tokenizers.Tokenizer.from_file(str(args.tokenizer))
    template=jinja2.Environment().from_string(read(args.tokenizer.with_name('tokenizer_config.json'))['chat_template'])
    for ref in read(ROOT.parent/'reference.json')['records']:
        actual=prepare('baseline',ref['request'],tokenizer,template)
        if actual['prompt']!=ref['prompt'] or actual['promptIds']!=ref['promptIds'] or actual['labelIds']!=ref['labelIds']:
            raise ValueError('old independent prompt/ID parity failed')
    rows=[]
    for variant in variants:
        for case,order,request in attempts(data):
            rows.append(dict(id=case['id'],order=order,variant=variant,request=request,
                             **prepare(variant,request,tokenizer,template)))
    result=dict(schema=1,datasetSha256=digest(args.dataset),promptsSha256=digest(ROOT/'prompts.py'),
                protocolSha256=digest(ROOT/'PROTOCOL.md'),modelSha256=MODEL,tokenizerSha256=TOKENIZER,
                llamaCommit=LLAMA,clientSha256=digest(args.client),clientSourceSha256=digest(ROOT/'reference.cpp'),
                environment=dict(platform=platform.platform(),backend='llama.cpp CPU',context=512,threads=4,
                                 kv='f16',flashAttention=False,tokenizers=tokenizers.__version__,jinja=jinja2.__version__),
                selectionSha256=digest(args.selection) if args.selection else None,
                variants={v:dict(model=v,loadMs=None,datasetSha256=digest(args.dataset),records=[]) for v in variants})
    # Preserve prepared requests before inference; failed/partial clients retain
    # their trace next to output. Report refuses an incomplete attempt population.
    requests=args.output.with_suffix('.requests.json')
    write(requests,dict(records=rows))
    with tempfile.TemporaryDirectory() as tmp:
        ids=Path(tmp)/'ids.txt'
        ids.write_text('\n'.join(' '.join(map(str,[len(r['labelIds']),*r['labelIds'],*r['promptIds']])) for r in rows)+'\n')
        raw=args.output.with_suffix('.raw.jsonl')
        subprocess.run([str(args.client),str(args.model),str(ids),str(raw)],check=True)
        values=[json.loads(line) for line in raw.read_text().splitlines()]
    if len(values)!=len(rows):
        raise ValueError('missing client attempts')
    for row,actual in zip(rows,values,strict=True):
        record=dict(row)
        if 'error' in actual:
            record.update(error=actual['error'],elapsedMs=actual.get('elapsedMs',0))
        else:
            if not 0<=actual['labelMass']<=1 or not math.isfinite(actual['labelMass']):
                raise ValueError('invalid label mass')
            record.update(actual)
            record['result']=score(actual['logits'],row['request']['choices'])
            for item in record['top']:
                item['text']=tokenizer.decode([item['tokenId']],skip_special_tokens=False)
            validate_result(record['result'],row['request']['choices'])
        result['variants'][row['variant']]['records'].append(record)
    write(args.output,result)
    if not args.selection:
        winner,summaries=selection(data,result['variants'],result['datasetSha256'])
        write(args.output.with_name('selection.json'),dict(winner=winner,promptsSha256=result['promptsSha256'],
              developmentSha256=result['datasetSha256'],runSha256=digest(args.output),summaries=summaries,
              rule='worst-order accuracy, pooled accuracy, lowest pooled Brier, fixed variant order'))
    else:
        summaries={v:describe(data,r,result['datasetSha256']) for v,r in result['variants'].items()}
    write(args.output.with_suffix('.summary.json'),summaries)
    for v,s in summaries.items():
        print(v,[x['accuracy'] for x in s['byOrder'].values()],s['labelMass'])


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['model','tokenizer','llama','client','dataset','output']:
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--selection',type=Path)
    run(p.parse_args())
