#!/usr/bin/env python3
"""Quantify original-weight Qwen differences from the preserved Q8 oracle."""
import argparse
import math
from common import ROOT, digest, load_dataset, pair_records, read, summarize, write


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('run'); p.add_argument('output')
    args=p.parse_args()
    run=read(args.run)
    if run['model']!='qwen3':
        raise ValueError('expected original-weight Qwen run')
    dataset=ROOT.parent/'cases.json'
    data=load_dataset(dataset)
    pairs=pair_records(data,run,digest(dataset))
    reference=read(ROOT.parent/'reference.json')
    oracle={(r['id'],r['order']):r for r in reference['records']}
    failures=matches=0
    logit_errors=[];score_errors=[];disagreements=[]
    for row,case,request in pairs:
        ref=oracle[row['id'],row['order']]
        if 'error' in row:
            failures+=1
            continue
        if row['tokenPaths']!=[ref['promptIds']]:
            raise ValueError('bridge prompt IDs differ')
        logits=ref['logits']
        es=[math.exp(z-max(logits)) for z in logits]
        probs=[v/sum(es) for v in es]
        choice=request['choices'][max(range(len(probs)),key=probs.__getitem__)]
        same=row['result']['choice']==choice
        matches+=same
        if not same:
            disagreements.append(dict(id=row['id'],order=row['order'],expected=case['expected'],
                                      originalWeight=row['result']['choice'],q8Reference=choice))
        for s,z,prob in zip(row['result']['scores'],logits,probs,strict=True):
            logit_errors.append(abs(s['logit']-z));score_errors.append(abs(s['score']-prob))
    result=dict(schema=1,runSha256=digest(args.run),referenceSha256=digest(ROOT.parent/'reference.json'),
                winningChoicesMatch=matches,attempts=len(pairs),failures=failures,
                maxLogitError=max(logit_errors,default=None),maxScoreError=max(score_errors,default=None),
                disagreements=disagreements,metrics=summarize(data,run,digest(dataset)),
                note='Original-weight PyTorch FP32 vs llama.cpp Q8 with f16 KV. Differences include precision and runtime; no parity tolerance or deployment equivalence claimed.')
    write(args.output,result)
    print({k:v for k,v in result.items() if k not in ['metrics','disagreements']})


if __name__=='__main__':
    main()
