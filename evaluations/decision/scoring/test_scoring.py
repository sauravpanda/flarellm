"""Offline invariants and hand-computed scoring/selection checks; no weights."""
import copy
import math
import unittest
from experiment import LLAMA, MODEL, TOKENIZER, ROOT, VARIANTS, attempts, digest, load_dataset, read, score, selection
from prompts import EXAMPLES, render


class FakeTemplate:
    def render(self, **kwargs):
        self.kwargs=kwargs
        return repr(kwargs['messages'])


class ScoringTests(unittest.TestCase):
    def assert_nested_equal(self, actual, expected):
        # Python versions differ in summation rounding; identities stay exact.
        if isinstance(expected, float):
            self.assertTrue(math.isclose(actual, expected, rel_tol=0, abs_tol=1e-12))
        elif isinstance(expected, dict):
            self.assertEqual(actual.keys(), expected.keys())
            for key in expected:
                with self.subTest(key=key):
                    self.assert_nested_equal(actual[key], expected[key])
        elif isinstance(expected, list):
            self.assertEqual(len(actual), len(expected))
            for a, b in zip(actual, expected, strict=True):
                self.assert_nested_equal(a, b)
        else:
            self.assertEqual(actual, expected)

    def test_recorded_selection_and_heldout_report(self):
        from report import build
        results=ROOT/'results'
        chosen=read(results/'selection.json')
        dev=read(results/'dev-results.json')
        dev_data=ROOT.parent/'comparison/dev.json'
        self.assertEqual(chosen['runSha256'],digest(results/'dev-results.json'))
        self.assertEqual(chosen['developmentSha256'],digest(dev_data))
        self.assertEqual(chosen['promptsSha256'],digest(ROOT/'prompts.py'))
        winner,summaries=selection(load_dataset(dev_data),dev['variants'],digest(dev_data))
        self.assertEqual(winner,chosen['winner'])
        self.assert_nested_equal(summaries,chosen['summaries'])
        self.assert_nested_equal(summaries,read(results/'dev-results.summary.json'))
        run=read(results/'test-results.json')
        for artifact in [dev,run]:
            self.assertEqual(artifact['modelSha256'],MODEL)
            self.assertEqual(artifact['tokenizerSha256'],TOKENIZER)
            self.assertEqual(artifact['llamaCommit'],LLAMA)
            self.assertEqual(artifact['promptsSha256'],digest(ROOT/'prompts.py'))
            self.assertEqual(artifact['protocolSha256'],digest(ROOT/'PROTOCOL.md'))
            self.assertEqual(artifact['clientSourceSha256'],digest(ROOT/'reference.cpp'))
        recorded=read(results/'report.json')
        self.assertEqual(run['selectionSha256'],digest(results/'selection.json'))
        self.assertEqual(run['datasetSha256'],digest(ROOT/'test.json'))
        self.assertEqual(recorded.pop('runSha256'),digest(results/'test-results.json'))
        self.assertEqual(recorded.pop('datasetSha256'),digest(ROOT/'test.json'))
        self.assertEqual(recorded.pop('selectionSha256'),digest(results/'selection.json'))
        self.assert_nested_equal(build(load_dataset(ROOT/'test.json'),run,chosen),recorded)
        self.assert_nested_equal(recorded['summaries'],read(results/'test-results.summary.json'))

    def test_recorded_flare_subset_and_numerical_audit(self):
        from verify_flare import audit, requests
        import tempfile
        from pathlib import Path
        results=ROOT/'results'
        wanted=read(results/'flare-requests.json')
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'requests.json'
            requests(read(results/'test-results.json'),path)
            self.assertEqual(read(path),wanted)
        actual=read(results/'flare-results.json')
        recorded=read(results/'flare-audit.json')
        self.assertEqual(recorded.pop('requestsSha256'),digest(results/'flare-requests.json'))
        self.assertEqual(recorded.pop('runSha256'),digest(results/'flare-results.json'))
        self.assert_nested_equal(audit(wanted,actual),recorded)
        bad=copy.deepcopy(actual)
        bad['records'][0]['promptIds'][0]+=1
        with self.assertRaises(ValueError):audit(wanted,bad)
        bad=copy.deepcopy(actual)
        for score in bad['records'][0]['result']['scores']:
            score['logit']+=10
        with self.assertRaises(ValueError):audit(wanted,bad)
        bad=copy.deepcopy(actual)
        bad['records'][0]['labelMass']=float('nan')
        with self.assertRaises(ValueError):audit(wanted,bad)
        with self.assertRaises(ValueError):audit(wanted,dict(records=actual['records'][:-1]))

    def test_frozen_inputs_disjoint_from_prior_cases(self):
        self.assertEqual(digest(ROOT/'prompts.py'),'1395fa029d71336e3d97f4481a35835c378f14cfe23eb345137fa5bbd66d9c49')
        self.assertEqual(digest(ROOT/'test.json'),'ec76803ebf51112c2ef750ddcf1335c09e1f60a3950e783d933a91c90b03223c')
        old=set(text for text,_ in EXAMPLES)
        for path in [ROOT.parent/'cases.json',ROOT.parent/'comparison/dev.json',ROOT.parent/'comparison/test.json']:
            old.update(c['state'] for c in read(path)['cases'])
        data=load_dataset(ROOT/'test.json')
        self.assertEqual(len(data['cases']),32)
        states=[c['state'] for c in data['cases']]
        self.assertEqual(len(set(states)),32)
        self.assertFalse(set(states)&old)
        for label in data['choices']:
            self.assertEqual(sum(c['expected']==label for c in data['cases']),8)
        for pos in range(4):
            self.assertEqual(sorted(order[pos] for order in data['orders']),list(range(4)))

    def test_examples_follow_permutation_not_fixed_letters(self):
        template=FakeTemplate()
        choices=['other','account','technical','billing']
        render('fewshot',dict(state='unseen',question='queue?',choices=choices),template)
        text=template.kwargs['messages'][0]['content']
        self.assertIn('Please reimburse an accidental purchase."\nAnswer: D',text)
        self.assertIn('I forgot the username needed to sign in."\nAnswer: B',text)
        self.assertFalse(template.kwargs['enable_thinking'])
        self.assertTrue(template.kwargs['add_generation_prompt'])

    def test_prefix_and_continuations(self):
        request=dict(state='ticket',question='queue?',choices=['a','b'])
        prompt,labels=render('answer_prefix',request,FakeTemplate())
        self.assertTrue(prompt.endswith('Answer:'))
        self.assertEqual(labels,[' A',' B'])
        with self.assertRaises(ValueError):render('unknown',request,FakeTemplate())
        with self.assertRaises(ValueError):render('fewshot',request,FakeTemplate())

    def test_stable_conditional_scores_and_ties(self):
        a=score([1000,1000],['a','b'])
        self.assertEqual(a['choice'],'a')
        self.assertEqual([s['score'] for s in a['scores']],[.5,.5])
        self.assertAlmostEqual(score([math.log(3),0],['a','b'])['scores'][0]['score'],.75)
        with self.assertRaises(ValueError):score([float('nan'),0],['a','b'])
        with self.assertRaises(ValueError):score([0],['a','b'])

    def test_selection_prefers_worst_order_and_fixed_ties(self):
        data=dict(choices=['a','b'],orders=[[0,1],[1,0]],question='q',cases=[dict(id='c',state='s',expected='a',kind='clear')])
        runs={}
        for v in VARIANTS:
            records=[]
            for case,order,request in attempts(data):
                logits=[2 if c=='a' else 0 for c in request['choices']]
                records.append(dict(id='c',order=order,elapsedMs=1,labelMass=.2,labelIds=[32,33],
                                    top=[dict(tokenId=32)],result=score(logits,request['choices'])))
            runs[v]=dict(model=v,loadMs=None,datasetSha256='sha',records=records)
        self.assertEqual(selection(data,runs,'sha')[0],'baseline')
        runs['baseline']['records'][1]['result']=score([2,0],['b','a'])
        self.assertEqual(selection(data,runs,'sha')[0],'concise')
        with self.assertRaises(ValueError):selection(data,{'baseline':runs['baseline']},'sha')
        missing=copy.deepcopy(runs);missing['concise']['records'].pop()
        with self.assertRaises(ValueError):selection(data,missing,'sha')


if __name__=='__main__':unittest.main()
