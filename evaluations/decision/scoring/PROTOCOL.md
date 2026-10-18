# Frozen Q8 scoring experiment

2026-10-18. This experiment follows #542. No production prompt is changed by the
experiment itself. Do not infer deployment readiness from synthetic accuracy.

1. Use the 16 published `comparison/dev.json` tickets for selection, with its
   four choice permutations. No training or calibration fitting. Compare exactly
   four candidates in `prompts.py`: unchanged baseline, concise user prompt,
   five independent examples, and concise prompt plus `Answer:` with leading-space
   letter continuations. These are the entire search budget; no adaptive prompt
   additions after seeing results. The few-shot examples are original, separately
   authored demonstrations, not development/test cases.
2. Use the same pinned Qwen3-0.6B Q8_0 and original tokenizer as #541. Evaluate
   with pinned llama.cpp CPU, four threads, 512 context, f16 KV, flash attention
   disabled. This is actual Q8 inference in an independent engine, not a claim
   that its runtime or all logits exactly match Flare. Verify baseline prompts
   against the old independent reference. Record all attempts, label logits,
   full-vocabulary probability mass on offered labels, and top-five next tokens.
   Scores used for predictions remain softmax conditional on offered labels.
3. Select lexicographically by highest worst-order development accuracy, highest
   pooled paired development accuracy, then lowest pooled Brier. Exact ties use
   the variant order in `prompts.py`. Save the winner and source/data/asset hashes
   before held-out inference. Always retain the unchanged baseline for comparison.
4. New held-out test: 32 original synthetic tickets, eight per class. The 24
   specific-queue cases are clear; `other` contains six out-of-scope and two
   ambiguous cases. Freeze file hashes before inference. A single author assigns
   targets under the existing routing policy; no independent human adjudication.
   Old published test results may motivate this experiment but are not selection
   inputs. No prompt changes after looking at the new test.
5. Run the winner and baseline on all 32 cases in all four orders (paired
   measurements, not 128 independent cases). Failures count incorrect and receive
   worst-case Brier 2. Report per-order and per-class results, five-bin ECE on
   completed rows, order changes, and label mass. Reject missing/duplicate rows.
6. Before considering a default change, the winner must beat 25% and the baseline
   in every test order, and exceed 50% for every class in original order. These
   are preliminary experiment gates, not deployment acceptance criteria.
7. Validate winner logits against actual Flare Q8 CPU on the first held-out case
   of each class in original and reversed orders (eight predeclared requests).
   Use the existing real-Qwen tolerance `0.8 + 0.02 * abs(reference)` for candidate
   logits and 0.15 absolute conditional-score difference; report winner agreement
   and all discrepancies. The diagnostic executable may accept explicit token IDs;
   it must stay an example, not a new unchecked application API.
8. Publish negative results too. Do not broaden the experiment or retune on this
   newly exposed test set. Follow-up choices are improving supervision/scoring if
   quality fails, or adopting a validated prompt and optimizing Q8 × f32 kernels
   if it passes. No NanoJev port, speed claim, or browser support claim from this PR.

Frozen input hashes:
- `prompts.py`: `1395fa029d71336e3d97f4481a35835c378f14cfe23eb345137fa5bbd66d9c49`
- `test.json`: `ec76803ebf51112c2ef750ddcf1335c09e1f60a3950e783d933a91c90b03223c`
- Development: `comparison/dev.json` from #542, unchanged.
