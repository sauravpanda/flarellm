#!/usr/bin/env python3
"""Report complete runs and paired model differences; no model dependencies."""
import argparse
from common import digest, load_dataset, pair_records, read, summarize, write


def compare(data, left, right, sha):
    a = pair_records(data, left, sha)
    b = pair_records(data, right, sha)
    paired = {}
    for order in range(len(data['orders'])):
        counts = dict(bothCorrect=0, qwenOnlyCorrect=0, nanojevOnlyCorrect=0, neitherCorrect=0)
        for (ra, case, _), (rb, other, _) in zip(a, b, strict=True):
            if (ra['id'],ra['order']) != (rb['id'],rb['order']) or case != other:
                raise ValueError('unpaired records')
            if ra['order'] != order:
                continue
            ca = ra.get('result', {}).get('choice') == case['expected']
            cb = rb.get('result', {}).get('choice') == case['expected']
            counts['bothCorrect' if ca and cb else 'qwenOnlyCorrect' if ca else 'nanojevOnlyCorrect' if cb else 'neitherCorrect'] += 1
        paired[str(order)] = counts
    return paired


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('dataset'); p.add_argument('qwen'); p.add_argument('nanojev'); p.add_argument('output')
    args = p.parse_args()
    data, sha = load_dataset(args.dataset), digest(args.dataset)
    qwen, nano = read(args.qwen), read(args.nanojev)
    if qwen['model'] != 'qwen3' or nano['model'] != 'nanojev':
        raise ValueError('expected Qwen then NanoJev')
    # Refuse a misleading mixed-hardware/backend latency comparison.
    if qwen['environment'] != nano['environment']:
        raise ValueError('runtime environments differ; report separately')
    result = dict(schema=1, datasetSha256=sha, inputSha256={'qwen3':digest(args.qwen), 'nanojev':digest(args.nanojev)},
                  models=[summarize(data, run, sha) for run in [qwen,nano]],
                  pairedByOrder=compare(data,qwen,nano,sha),
                  caveats=['Original synthetic English data with author-assigned targets; no real-world accuracy claim.',
                           'Orders are repeated paired measurements, not independent examples.',
                           'ECE is completed-only; failures count incorrect and receive Brier 2.',
                           'Scores are uncalibrated. Other is an ordinary candidate, not an abstention threshold.',
                           'PyTorch CPU float32 comparison; not Flare Q8, browser, CUDA or published NanoJev BF16 performance.'])
    write(args.output,result)
    for model in result['models']:
        print(model['model'], [v['accuracy'] for v in model['byOrder'].values()],model['orderSensitivity'],model['latency'])


if __name__ == '__main__':
    main()
