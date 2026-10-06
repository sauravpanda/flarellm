"""Validate independent reference parity and summarize every held-out attempt.
Usage: report.py REFERENCE.json RUN.json OUTPUT.json
RUN is decision_eval output or browser run.mjs output (the decisions member).
"""
import hashlib,json,math,statistics,sys
from pathlib import Path
reference=json.loads(Path(sys.argv[1]).read_text())
source=json.loads(Path(sys.argv[2]).read_text())
run=source.get('decisions',source)
records=run['records']; expected=reference['records']
assert len(records)==len(expected),'Missing attempts must not be filtered from a report'
def softmax(logits):
    es=[math.exp(x-max(logits)) for x in logits];return [x/sum(es) for x in es]
def metrics(rows):
    correct=0;brier=0;bins=[[] for _ in range(5)];failures=0;overflow=0
    for actual,ref in rows:
        result=actual.get('result')
        if not result:
            failures+=1;overflow+=int('context' in actual.get('error','').lower());brier+=2;continue
        good=result['choice']==ref['expected'];correct+=good
        probabilities=[s['score'] for s in result['scores']]
        brier+=sum((p-int(c==ref['expected']))**2 for p,c in zip(probabilities,ref['request']['choices']))
        confidence=max(probabilities);bins[min(4,int(confidence*5))].append((confidence,int(good)))
    n=len(rows);completed=n-failures
    ece=sum(len(b)*abs(statistics.mean(x[0] for x in b)-statistics.mean(x[1] for x in b)) for b in bins if b)/completed if completed else None
    return dict(n=n,correct=correct,accuracy=correct/n,failures=failures,failureRate=failures/n,overflowRate=overflow/n,brier=brier/n,ece5Completed=ece,completed=completed)
pairs=list(zip(records,expected,strict=True));max_logit=max_score=0;winner_matches=0;compact=[]
for actual,ref in pairs:
    assert (actual['id'],actual['order'])==(ref['id'],ref['order'])
    result=actual.get('result')
    if result:
        assert result['prompt']==ref['prompt'] and result['promptIds']==ref['promptIds']
        assert result['promptVersion']==reference['promptVersion']
        assert [s['choice'] for s in result['scores']]==ref['request']['choices']
        assert [s['tokenId'] for s in result['scores']]==ref['labelIds']
        assert result['choice']==ref['request']['choices'][result['index']]
        assert all(math.isfinite(s['score']) and math.isfinite(s['logit']) for s in result['scores'])
        assert abs(sum(s['score'] for s in result['scores'])-1)<1e-12
        oracle=softmax(ref['logits']);winner_matches+=result['index']==max(range(len(oracle)),key=lambda i:oracle[i])
        for score,logit,prob in zip(result['scores'],ref['logits'],oracle):
            le=abs(score['logit']-logit);se=abs(score['score']-prob);max_logit=max(max_logit,le);max_score=max(max_score,se)
            assert le<=reference['logitTolerance']['absolute']+reference['logitTolerance']['relative']*abs(logit),(ref['id'],le)
            assert se<=reference['scoreAbsoluteTolerance'],(ref['id'],se)
    compact.append({**{k:v for k,v in actual.items() if k!='result'},**({'result':{k:v for k,v in result.items() if k not in ('prompt','promptIds')}} if result else {})})
base=[p for p in pairs if p[1]['order']==0];reverse=[p for p in pairs if p[1]['order']==1]
changed=sum(a.get('result',{}).get('choice')!=b.get('result',{}).get('choice') for (a,_),(b,_) in zip(base,reverse,strict=True))
latencies=[r['elapsedMs'] for r in records]
report=dict(schema=1,referenceSha256=hashlib.sha256(Path(sys.argv[1]).read_bytes()).hexdigest(),backend=run['backend'],browser=source.get('browser'),wasmCapacityBytes=source.get('decisionWasmCapacityBytes'),loadMs=run['loadMs'],baseline=metrics(base),reversed=metrics(reverse),byKind={k:metrics([p for p in base if p[1]['kind']==k]) for k in ['clear','ambiguous','none-of-the-above']},optionOrder=dict(changed=changed,n=len(base),fraction=changed/len(base),note='Reversal changes both option order and label assignment; their effects are not separately identified.'),referenceParity=dict(maxLogitError=max_logit,maxScoreError=max_score,winnerMatches=winner_matches,attempts=len(records)),latencyMs=dict(first=latencies[0],warmMedian=statistics.median(latencies[1:]),warmMin=min(latencies[1:]),warmMax=max(latencies[1:])),caveats=['Synthetic n=16, repeated in two orders; not 32 independent examples or real-world accuracy.','Multiclass Brier is sum over classes (0..2); failures receive worst-case 2 and count incorrect.','ECE uses five equal-width bins on completed predictions only; report coverage alongside it. No calibration fitted.','No quality tuning on held-out cases. Other is an ordinary candidate, not a confidence threshold.'],records=compact)
Path(sys.argv[3]).write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k!='records'},indent=2))
