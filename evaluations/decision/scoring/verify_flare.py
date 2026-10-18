#!/usr/bin/env python3
"""Predeclared Flare Q8 smoke subset and independent-engine numerical audit."""
import argparse
import math
from experiment import ROOT, digest, load_dataset, read, validate_result, write


def requests(reference, output):
    data=load_dataset(ROOT/'test.json')
    if reference['datasetSha256']!=digest(ROOT/'test.json'):
        raise ValueError('held-out source mismatch')
    chosen=read(ROOT/'results/selection.json')['winner']
    first={}
    for case in data['cases']:
        first.setdefault(case['expected'],case['id'])
    rows=[r for r in reference['variants'][chosen]['records'] if r['id'] in first.values() and r['order'] in [0,1]]
    if len(rows)!=8:
        raise ValueError('missing predeclared smoke cases')
    write(output,dict(referenceSha256=digest(ROOT/'results/test-results.json'),records=rows))


def audit(wanted, actual):
    if len(actual['records'])!=len(wanted['records']) or len(actual['records'])!=8:
        raise ValueError('missing smoke attempts')
    max_logit=max_score=max_mass=0
    winners=0; disagreements=[]
    for ref,row in zip(wanted['records'],actual['records'],strict=True):
        for key in ['id','order','variant','promptIds','labelIds']:
            if row[key]!=ref[key]:raise ValueError('smoke input mismatch: '+key)
        validate_result(row['result'],ref['request']['choices'])
        for value in [row['labelMass'],ref['labelMass']]:
            if not math.isfinite(value) or not 0<=value<=1:
                raise ValueError('invalid full-vocabulary label mass')
        matched=row['result']['choice']==ref['result']['choice'];winners+=matched
        if not matched:disagreements.append(dict(id=row['id'],order=row['order'],flare=row['result']['choice'],reference=ref['result']['choice']))
        for a,b in zip(row['result']['scores'],ref['result']['scores'],strict=True):
            le=abs(a['logit']-b['logit']);se=abs(a['score']-b['score'])
            max_logit=max(max_logit,le);max_score=max(max_score,se)
            if le>.8+.02*abs(b['logit']) or se>.15:
                raise ValueError(f'smoke numerical tolerance exceeded: {row["id"]} {le} {se}')
        max_mass=max(max_mass,abs(row['labelMass']-ref['labelMass']))
    return dict(attempts=8,winnerMatches=winners,maxCandidateLogitError=max_logit,
                maxConditionalScoreError=max_score,maxLabelMassError=max_mass,disagreements=disagreements,
                note='Predeclared eight-case numerical smoke only; full test quality measured with llama.cpp Q8. No browser or speed claim.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['requests','audit']);p.add_argument('input');p.add_argument('output');p.add_argument('--actual')
    args=p.parse_args()
    if args.mode=='requests':requests(read(args.input),args.output)
    else:
        if not args.actual:p.error('--actual required')
        result=audit(read(args.input),read(args.actual))
        result.update(requestsSha256=digest(args.input),runSha256=digest(args.actual))
        write(args.output,result);print(result)
