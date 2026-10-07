#!/usr/bin/env python3
"""Explicit opt-in public asset download. No authentication; verify all hashes."""
import argparse
from pathlib import Path
from common import ROOT, digest, read


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('destination', type=Path)
    args = p.parse_args()
    from huggingface_hub import snapshot_download
    for model, pin in read(ROOT/'assets.json').items():
        destination = args.destination/model
        snapshot_download(pin['repository'], revision=pin['revision'], token=False,
                          local_dir=destination, allow_patterns=list(pin['files']))
        for name, sha in pin['files'].items():
            if digest(destination/name) != sha:
                raise ValueError(f'checksum mismatch: {model}/{name}')
        print(f'Verified {model}: {destination}')


if __name__ == '__main__':
    main()
