#!/usr/bin/env python3
"""Opt-in CPU adapter check against the unmodified upstream predict method."""
import argparse
import os
from pathlib import Path
from common import ROOT, digest, load_dataset, nanojev_request, write
from run import Engine, verify_assets


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('assets',type=Path);p.add_argument('output',type=Path)
    args=p.parse_args()
    verify_assets('nanojev',args.assets)
    os.environ['HF_HUB_OFFLINE']=os.environ['TRANSFORMERS_OFFLINE']='1'
    import torch
    torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.manual_seed(0)
    engine=Engine('nanojev',args.assets)
    # Bypass only the constructor's CUDA guard. predict itself is unchanged and
    # uses disabled CUDA autocast for fp32, so its operations follow CPU tensors.
    stock=engine.upstream.DecisionPredictor.__new__(engine.upstream.DecisionPredictor)
    stock.model=engine.model;stock.tokenizer=engine.tokenizer
    stock.root=args.assets;stock.run_config={'model':'Qwen/Qwen3-0.6B','set_head':'attention'}
    stock.limit=512;stock.device=torch.device('cpu');stock.precision='fp32'
    stock.disable_native_triton=False;stock.inference_calls=0;stock._torch=torch
    data=load_dataset(ROOT/'dev.json')
    checks=[]
    for order in [data['choices'],list(reversed(data['choices']))]:
        request=dict(state=data['cases'][0]['state'],question=data['question'],choices=order)
        actual=engine.decide(request)
        upstream=stock.predict(nanojev_request(request))['states'][0]['answers']['queue']
        if actual['result']['choice']!=upstream['choice']:
            raise ValueError('upstream winner mismatch')
        errors=[abs(s['score']-upstream['probabilities'][s['choice']]) for s in actual['result']['scores']]
        if max(errors)>1e-7:
            raise ValueError('upstream score mismatch')
        checks.append(dict(choices=order,maxScoreError=max(errors),choice=upstream['choice']))
    write(args.output,dict(schema=1,sourceSha256=digest(__file__),checks=checks,
                           note='Real checkpoint, CPU FP32. Unchanged upstream predict method with CPU-initialized fields; no CUDA/BF16 parity claim.'))


if __name__=='__main__':
    main()
