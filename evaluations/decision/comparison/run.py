#!/usr/bin/env python3
"""Offline, opt-in CPU comparison; supplied local assets must match assets.json."""
import argparse
import importlib.metadata
import importlib.util
import os
import platform
import resource
import sys
import time
from pathlib import Path

from common import (ROOT, attempts, digest, load_dataset, nanojev_request,
                    qwen_messages, read, validate_result, write)


class ContextOverflow(ValueError):
    pass


def verify_assets(kind, root):
    pin = read(ROOT / 'assets.json')[kind]
    for name, expected in pin['files'].items():
        if digest(root / name) != expected:
            raise ValueError(f'asset checksum mismatch: {name}')
    return pin


def import_upstream(root):
    spec = importlib.util.spec_from_file_location('nanojev_pinned_predictor', root / 'source/scripts/predict_toy_decisions.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_qwen_prompt(tokenizer):
    # Full original-tokenizer and official-template parity with the independent
    # llama.cpp baseline; not merely comparing two copies of our new renderer.
    reference = read(ROOT.parent / 'reference.json')
    for row in reference['records']:
        prompt = tokenizer.apply_chat_template(qwen_messages(row['request']), tokenize=False,
                                               add_generation_prompt=True, enable_thinking=False)
        ids = tokenizer.encode(prompt, add_special_tokens=False)
        if prompt != row['prompt'] or ids != row['promptIds']:
            raise ValueError('qwen3-choice-v1 independent prompt/ID parity failed')
        for i in range(len(row['request']['choices'])):
            if tokenizer.encode(prompt + chr(65+i), add_special_tokens=False) != ids + [32+i]:
                raise ValueError('answer-boundary label parity failed')


class Engine:
    def __init__(self, kind, root):
        import torch
        from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoTokenizer
        self.torch, self.kind = torch, kind
        if kind == 'qwen3':
            self.tokenizer = AutoTokenizer.from_pretrained(root, local_files_only=True, trust_remote_code=False)
            verify_qwen_prompt(self.tokenizer)
            self.model = AutoModelForCausalLM.from_pretrained(
                root, local_files_only=True, trust_remote_code=False,
                dtype=torch.float32, attn_implementation='sdpa').eval()
            self.upstream = None
        else:
            from safetensors.torch import load_file
            self.upstream = import_upstream(root)
            self.tokenizer = AutoTokenizer.from_pretrained(root / 'tokenizer', local_files_only=True,
                                                           trust_remote_code=False)
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            config = AutoConfig.from_pretrained(root / 'backbone_config', local_files_only=True,
                                                 trust_remote_code=False)
            config.use_cache = False
            body = AutoModel.from_config(config, attn_implementation='sdpa', trust_remote_code=False).float()
            # Unmodified upstream architecture, strict full weights; only device
            # initialization differs from the CUDA-only DecisionPredictor.
            self.model = self.upstream.load_decision_model_class()(body, read(root / 'config.json')['set_head'])
            weights = load_file(str(root / 'best.safetensors'), device='cpu')
            self.model.load_state_dict(weights, strict=True)
            del weights
            self.model.float().eval()
        if any(p.device.type != 'cpu' or p.dtype != torch.float32 for p in self.model.parameters()):
            raise ValueError('comparison requires CPU float32 parameters')

    def decide(self, request):
        torch = self.torch
        prep_start = time.perf_counter()
        if self.kind == 'qwen3':
            prompt = self.tokenizer.apply_chat_template(qwen_messages(request), tokenize=False,
                                                        add_generation_prompt=True, enable_thinking=False)
            ids = self.tokenizer.encode(prompt, add_special_tokens=False)
            for i in range(len(request['choices'])):
                if self.tokenizer.encode(prompt + chr(65+i), add_special_tokens=False) != ids + [32+i]:
                    raise ValueError('answer label does not append one token')
            if len(ids) > 512:
                raise ContextOverflow('Qwen prompt exceeds 512 tokens; no truncation')
            tokens = torch.tensor([ids], dtype=torch.long)
            paths = [ids]
        else:
            payload = nanojev_request(request)
            # Upstream rejects overflow without truncating; use the shared 512
            # limit instead of the checkpoint's larger training maximum.
            try:
                examples = self.upstream.prepare_examples(payload, self.tokenizer, 512)
            except ValueError as exc:
                if 'max_length' in str(exc):
                    raise ContextOverflow(str(exc)) from exc
                raise
            paths = examples[0]['leaf_tokens']
        prepared = time.perf_counter()
        with torch.inference_mode():
            if self.kind == 'qwen3':
                hidden = self.model.model(input_ids=tokens, attention_mask=torch.ones_like(tokens),
                                          use_cache=False).last_hidden_state[0, -1]
                # Exactly the corresponding LM-head rows, not generation or a
                # vocabulary probability. Avoid materializing all prompt logits.
                logits = torch.nn.functional.linear(hidden, self.model.lm_head.weight[32:32+len(request['choices'])])
            else:
                logits, _ = self.model(examples, self.tokenizer.pad_token_id)
                logits = logits[0, :len(request['choices'])]
            if not torch.isfinite(logits).all():
                raise ValueError('nonfinite logits')
            probs = logits.float().softmax(-1).tolist()
        finished = time.perf_counter()
        winner = max(range(len(probs)), key=probs.__getitem__)
        if self.upstream:
            answer = self.upstream.answer_from_probabilities(examples[0], probs)
            if answer['choice'] != request['choices'][winner] or list(answer['probabilities'].values()) != probs:
                raise ValueError('upstream answer mapping disagrees')
        result = dict(choice=request['choices'][winner], scores=[
            dict(choice=c, logit=z, score=p) for c,z,p in zip(request['choices'], logits.tolist(), probs, strict=True)])
        validate_result(result, request['choices'])
        return dict(result=result, preparationMs=(prepared-prep_start)*1000,
                    forwardAndScoreMs=(finished-prepared)*1000,
                    pathLengths=list(map(len, paths)), tokenPaths=paths)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=['qwen3', 'nanojev'], required=True)
    parser.add_argument('--assets', type=Path, required=True)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    data = load_dataset(args.dataset)
    pin = verify_assets(args.model, args.assets)
    for key in ['HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'HF_HUB_DISABLE_TELEMETRY']:
        os.environ[key] = '1'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    import torch
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.manual_seed(0)
    started = time.perf_counter()
    engine = Engine(args.model, args.assets)
    load_ms = (time.perf_counter()-started)*1000
    run = dict(schema=1, model=args.model, datasetSha256=digest(args.dataset),
               assetPin=pin, sourceSha256={name: digest(ROOT/name) for name in ['run.py', 'common.py']},
               environment=dict(platform=platform.platform(), machine=platform.machine(), python=sys.version,
                                packages={name: importlib.metadata.version(name) for name in
                                          ['torch','transformers','tokenizers','safetensors','huggingface-hub']},
                                torchConfig=torch.__config__.show(), device='cpu', dtype='float32',
                                intraOpThreads=4, interOpThreads=1, questionsPerCall=1,
                                attention='sdpa', context=512, temperature=1, kvReuse=False),
               loadMs=load_ms, records=[])
    for case, order, request in attempts(data):
        started = time.perf_counter()
        row = dict(id=case['id'], order=order)
        try:
            row.update(engine.decide(request))
        except Exception as exc:
            row.update(errorType=type(exc).__name__, error=str(exc))
        row['elapsedMs'] = (time.perf_counter()-started)*1000
        run['records'].append(row)
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        run['peakRssBytes'] = peak if sys.platform == 'darwin' else peak*1024
        run['peakRssScope'] = 'process high-water mark including imports, asset verification and model load'
        write(args.output, run)  # Interrupted runs retain evidence; report rejects missing attempts.
        print(f"{args.model} {case['id']} order {order}: {row['elapsedMs']:.1f} ms "
              f"{row.get('result', {}).get('choice', row.get('error'))}", flush=True)
    return int(any('error' in row for row in run['records']))


if __name__ == '__main__':
    raise SystemExit(main())
