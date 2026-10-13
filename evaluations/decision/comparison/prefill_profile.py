#!/usr/bin/env python3
"""Prepare fixed development requests or report native --profile measurements."""
import argparse
import statistics
from common import ROOT, attempts, digest, load_dataset, read, validate_result, write


def requests(output):
    data=load_dataset(ROOT/'dev.json')
    first={}
    for case,order,request in attempts(data):
        if order==0:
            first.setdefault(case['expected'],dict(id=case['id'],request=request))
    rows=[dict(**row,order=repeat) for repeat in range(2) for row in first.values()]
    write(output,dict(datasetSha256=digest(ROOT/'dev.json'),records=rows,
                      note='First development ticket per class, original choices, two repetitions; order denotes repetition here.'))


def report(request_path, run_path, unprofiled_path, output):
    wanted,run,plain=read(request_path),read(run_path),read(unprofiled_path)
    if len(run['records'])!=len(wanted['records']) or len(plain['records'])!=1:
        raise ValueError('missing profile/control attempts')
    for actual,expected in zip(run['records'],wanted['records'],strict=True):
        if (actual['id'],actual['order'])!=(expected['id'],expected['order']):
            raise ValueError('profile identity mismatch')
        validate_result(actual['result'],expected['request']['choices'])
        p=actual['prefillProfile']
        if p['seqLen']!=len(actual['result']['promptIds']) or p['totalMs']<=0:
            raise ValueError('missing prefill measurements')
    if run['records'][0]['result']!=plain['records'][0]['result']:
        raise ValueError('profiling changed decision result')
    profiles=[r['prefillProfile'] for r in run['records']]
    phases={key:statistics.mean(p[key] for p in profiles) for key in profiles[0] if key.endswith('Ms')}
    projection=sum(phases[key] for key in ['qkvProjMs','attnOutProjMs','gateUpMs','downMs'])
    write(output,dict(schema=1,requestsSha256=digest(request_path),runSha256=digest(run_path),
                      unprofiledSha256=digest(unprofiled_path),attempts=len(profiles),
                      profiledControlExactlyMatches=True,meanMs=phases,
                      packedProjectionShareOfPrefill=projection/phases['totalMs'],
                      warmMedianDecisionMs=statistics.median(r['elapsedMs'] for r in run['records'][1:]),
                      note='Native Q8 CPU phase timings, four distinct development tickets repeated twice; no browser or stable performance guarantee.'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='command',required=True)
    prepare=sub.add_parser('requests');prepare.add_argument('output')
    summarize=sub.add_parser('report')
    for key in ['requests','run','unprofiled','output']: summarize.add_argument(key)
    args=p.parse_args()
    if args.command=='requests':requests(args.output)
    else:report(args.requests,args.run,args.unprofiled,args.output)


if __name__=='__main__':
    main()
