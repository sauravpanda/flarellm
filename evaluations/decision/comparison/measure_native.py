#!/usr/bin/env python3
"""Run decision_eval in a fresh child and record its process high-water RSS."""
import argparse
import platform
import resource
import subprocess
import sys
from pathlib import Path
from common import digest, read, write


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['binary','model','tokenizer','requests','output']:
        p.add_argument(name,type=Path)
    p.add_argument('--profile',action='store_true')
    args=p.parse_args()
    command=[str(path.resolve()) for path in [args.binary,args.model,args.tokenizer,args.requests,args.output]]
    if args.profile:command.append('--profile')
    subprocess.run(command,check=True)
    run=read(args.output)
    peak=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    run['peakRssBytes']=peak if sys.platform=='darwin' else peak*1024
    run['peakRssScope']='fresh decision_eval child high-water mark, including model load'
    run['environment']=dict(platform=platform.platform(),machine=platform.machine(),
                            rust=subprocess.check_output(['rustc','--version'],text=True).strip())
    run['inputs']={name:digest(getattr(args,name)) for name in ['binary','model','tokenizer','requests']}
    write(args.output,run)


if __name__=='__main__':
    main()
