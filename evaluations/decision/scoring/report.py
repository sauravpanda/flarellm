#!/usr/bin/env python3
"""Quality gates and paired outcomes from the frozen held-out run."""
import argparse
from experiment import ROOT, describe, digest, load_dataset, read, write


def build(data, run, chosen):
    winner=chosen['winner'];sha=run['datasetSha256']
    expected=set(['baseline',winner])
    if set(run['variants'])!=expected or run['promptsSha256']!=chosen['promptsSha256']:
        raise ValueError('test variants/prompts differ from frozen selection')
    summaries={v:describe(data,r,sha) for v,r in run['variants'].items()}
    base=summaries['baseline'];selected=summaries[winner]
    classes={label:counts[label]/sum(counts.values())
             for label,counts in selected['byOrder']['0']['confusion'].items()}
    beats_baseline=all(selected['byOrder'][o]['accuracy']>base['byOrder'][o]['accuracy'] for o in base['byOrder'])
    above_chance=all(s['accuracy']>.25 for s in selected['byOrder'].values())
    class_gate=all(x>.5 for x in classes.values())
    baseline={(r['id'],r['order']):r for r in run['variants']['baseline']['records']}
    targets={c['id']:c['expected'] for c in data['cases']}
    paired={str(i):dict(bothCorrect=0,baselineOnly=0,selectedOnly=0,neitherCorrect=0) for i in range(len(data['orders']))}
    for r in run['variants'][winner]['records']:
        old=baseline[r['id'],r['order']];target=targets[r['id']]
        a=old.get('result',{}).get('choice')==target;b=r.get('result',{}).get('choice')==target
        key='bothCorrect' if a and b else 'baselineOnly' if a else 'selectedOnly' if b else 'neitherCorrect'
        paired[str(r['order'])][key]+=1
    return dict(selectedVariant=winner,uniqueTestCases=len(data['cases']),summaries=summaries,
                originalOrderClassAccuracy=classes,pairedByOrder=paired,
                preliminaryGate=dict(passed=beats_baseline and above_chance and class_gate,
                   beatsBaselineEveryOrder=beats_baseline,beats25PercentEveryOrder=above_chance,
                   eachOriginalOrderClassAbove50Percent=class_gate),
                caveats=['Synthetic n=32 with author-assigned targets; no independent human adjudication.',
                         'Choice orders are paired measurements, not independent additional cases.',
                         'Conditional scores and label mass are different quantities; neither is calibrated confidence.',
                         'Full quality run uses llama.cpp Q8 CPU; Flare is checked on a predeclared eight-case subset.',
                         'Passing these preliminary gates is not deployment acceptance.'])


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('dataset');p.add_argument('run');p.add_argument('selection');p.add_argument('output')
    args=p.parse_args()
    data=load_dataset(args.dataset);run=read(args.run);chosen=read(args.selection)
    if run['datasetSha256']!=digest(args.dataset) or run['selectionSha256']!=digest(args.selection):
        raise ValueError('input provenance mismatch')
    result=build(data,run,chosen)
    result.update(datasetSha256=digest(args.dataset),runSha256=digest(args.run),selectionSha256=digest(args.selection))
    write(args.output,result)
    print(result['selectedVariant'],result['preliminaryGate'],result['originalOrderClassAccuracy'])
