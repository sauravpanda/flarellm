"""No weights, downloads, PyTorch or tokenizers needed for these audit tests."""
import copy
import json
import tempfile
from pathlib import Path
import unittest
from common import (ROOT, attempts, digest, load_dataset, metrics, nanojev_request,
                    pair_records, qwen_messages, read, summarize, validate_result, write)


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.data = dict(choices=['a','b'], orders=[[0,1],[1,0]],
                         question='Which?', cases=[dict(id='one', state='s', expected='a',kind='clear')])
        self.run = dict(model='test',datasetSha256='hash',loadMs=0,records=[])
        for case, order, request in attempts(self.data):
            scores = [dict(choice=c,score=.8 if c=='a' else .2,logit=1 if c=='a' else 0) for c in request['choices']]
            self.run['records'].append(dict(id=case['id'],order=order,elapsedMs=10+order,
                                            result=dict(choice='a',scores=scores)))

    def test_frozen_splits_and_balanced_positions(self):
        seen = {c['state'].casefold() for c in read(ROOT.parent/'cases.json')['cases']}
        for split,n,sha in [('dev',16,'af00a4e161064b1b39ec6fddbc778feb1d9e32719f765b589e22cc29ffd4ed8d'),
                            ('test',64,'c3b797790eaf8c1efcae4a4609ba2cef3a0ddfea2e99c575808e586f3293867f')]:
            self.assertEqual(digest(ROOT/f'{split}.json'),sha)
            data=load_dataset(ROOT/f'{split}.json')
            self.assertEqual(len(data['cases']),n)
            for c in data['cases']:
                self.assertNotIn(c['state'].casefold(),seen)
                seen.add(c['state'].casefold())
            for label in data['choices']:
                self.assertEqual(sum(c['expected']==label for c in data['cases']),n//4)
            for pos in range(4):
                self.assertEqual(sorted(order[pos] for order in data['orders']),list(range(4)))

    def test_shared_semantics_without_gold(self):
        _,_,request=next(attempts(self.data))
        request['state']='A quoted "ticket"\nand another line'
        messages=qwen_messages(request)
        self.assertIn(json.dumps(request['state']),messages[1]['content'])
        nano=nanojev_request(request)['states'][0]
        self.assertEqual(nano['state'],request['state'])
        question=nano['questions']['queue']
        self.assertEqual(question['instructions'],request['question'])
        self.assertEqual(question['criteria'],{'a':'a','b':'b'})
        self.assertNotIn('expected',json.dumps(nano))

    def test_missing_duplicate_unknown_attempts_and_wrong_dataset_rejected(self):
        for mutation in ['missing','duplicate','unknown','hash']:
            run=copy.deepcopy(self.run)
            if mutation=='missing': run['records'].pop()
            if mutation=='duplicate': run['records'].append(run['records'][0])
            if mutation=='unknown': run['records'][0]['id']='elsewhere'
            if mutation=='hash': run['datasetSha256']='different'
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                pair_records(self.data,run,'hash')

    def test_probabilities_identity_and_winner_validated(self):
        for mutation in ['nan','negative','sum','winner','labels','logit']:
            result=copy.deepcopy(self.run['records'][0]['result'])
            if mutation=='nan': result['scores'][0]['score']=float('nan')
            if mutation=='negative': result['scores'][0]['score']=-.1
            if mutation=='sum': result['scores'][0]['score']=.3
            if mutation=='winner': result['choice']='b'
            if mutation=='labels': result['scores'].reverse()
            if mutation=='logit': result['scores'][0]['logit']=float('inf')
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                validate_result(result,['a','b'])

    def test_known_metrics_failures_and_completed_order_denominator(self):
        rows=pair_records(self.data,self.run,'hash')
        result=metrics(rows,['a','b'])
        self.assertEqual(result['accuracy'],1)
        self.assertAlmostEqual(result['brier'],.08)
        self.assertAlmostEqual(result['ece5Completed'],.2)
        row=self.run['records'][1]
        del row['result']
        row.update(error='too long',errorType='ContextOverflow')
        summary=summarize(self.data,self.run,'hash')
        self.assertEqual(summary['pooledPaired']['accuracy'],.5)
        self.assertAlmostEqual(summary['pooledPaired']['brier'],1.04)
        self.assertEqual(summary['pooledPaired']['overflow'],1)
        self.assertEqual(summary['orderSensitivity']['completeCases'],0)
        self.assertIsNone(summary['orderSensitivity']['fraction'])
        self.assertIsNone(summary['latency']['warmMedianMs'])

    def test_reordered_run_retains_identity_and_execution_timing(self):
        self.run['records'].reverse()
        summary=summarize(self.data,self.run,'hash')
        self.assertEqual(summary['byOrder']['0']['correct'],1)
        self.assertEqual(summary['latency']['firstAttemptMs'],11)
        self.assertEqual(summary['latency']['warmMedianMs'],10)

    def test_recorded_results_reproduce_summaries(self):
        # Committed real-model outputs are audited without loading either model.
        for split in ['dev', 'test']:
            data=load_dataset(ROOT/f'{split}.json')
            path=ROOT/'results'/f'{split}-summary.json'
            if not path.exists():
                self.fail(f'missing committed report: {path}')
            report=read(path)
            for model, recorded in zip(['qwen3','nanojev'],report['models'],strict=True):
                run_path=ROOT/'results'/f'{model}-{split}.json'
                self.assertEqual(report['inputSha256'][model],digest(run_path))
                self.assertEqual(recorded,summarize(data,read(run_path),digest(ROOT/f'{split}.json')))

    def test_artifact_formatting_preserves_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'trace.json'
            value={'tokens': [[1,2,-3],[4]], 'text':'[\n123\n]', 'score':.3}
            write(path,value)
            self.assertEqual(read(path),value)
            self.assertIn('[1,2,-3]',path.read_text())

    def test_native_profile_and_control_provenance(self):
        from prefill_profile import report
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp)/'profile-summary.json'
            report(ROOT/'profile-requests.json',ROOT/'results/flare-profile.json',
                   ROOT/'results/flare-unprofiled.json',output)
            self.assertEqual(read(output),read(ROOT/'results/profile-summary.json'))
        requests=read(ROOT/'profile-requests.json')
        self.assertEqual(requests['datasetSha256'],digest(ROOT/'dev.json'))
        self.assertEqual(read(ROOT/'profile-control-request.json')['records'],requests['records'][:1])

    def test_all_failure_is_not_calibrated_or_order_stable(self):
        for row in self.run['records']:
            del row['result']
            row.update(error='failed',errorType='RuntimeError')
        report=summarize(self.data,self.run,'hash')
        self.assertEqual(report['pooledPaired']['brier'],2)
        self.assertEqual(report['pooledPaired']['accuracy'],0)
        self.assertIsNone(report['pooledPaired']['ece5Completed'])
        self.assertEqual(report['latency']['completed'],0)
        self.assertIsNone(report['orderSensitivity']['fraction'])


if __name__=='__main__':
    unittest.main()
