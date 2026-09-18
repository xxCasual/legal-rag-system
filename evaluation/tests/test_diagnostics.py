import contextlib
import io
import math
import sys
import tempfile
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

class SummaryRegression(unittest.TestCase):
    def test_existing_summary_exposes_missing_denominator(self):
        import pandas as pd
        from run_evaluation import print_summary
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            print_summary(pd.DataFrame({'answer_relevancy':[0.8,math.nan]}), 'repro')
        self.assertIn('1/2', out.getvalue())
        self.assertIn('50.0%', out.getvalue())

    def test_retry_nan_and_input_failure_keep_every_row(self):
        from diagnostics import score_samples, summarize, read_replay_csv
        samples=[dict(user_input=str(i),response='answer',reference='ref',retrieved_contexts=['ctx']) for i in range(5)]
        samples[4]['response']=''
        def scorer(sample, metric, attempt):
            if sample['user_input']=='1' and attempt==1: raise TimeoutError('private response must not be persisted')
            if sample['user_input']=='2': return math.nan
            if sample['user_input']=='3': raise ValueError('bad response')
            return 0.0 if sample['user_input']=='0' else 0.8
        with tempfile.TemporaryDirectory() as d:
            import json
            path=Path(d)/'attempts.jsonl'
            rows=score_samples(samples,['answer_relevancy'],scorer,attempts=2,audit_path=path)
            audit=[json.loads(s) for s in path.read_text().splitlines()]
            self.assertEqual(len(rows),5)
            self.assertEqual(len(audit),8)
            self.assertNotIn('private response',path.read_text())
            self.assertEqual(rows[1]['answer_relevancy_status'],'ok_after_retry')
            self.assertEqual(rows[2]['answer_relevancy_status'],'non_finite_score')
            self.assertEqual(rows[3]['answer_relevancy_status'],'evaluator_exception')
            self.assertEqual(rows[4]['answer_relevancy_status'],'input_invalid')
            stat=summarize(rows,['answer_relevancy'])['answer_relevancy']
            self.assertEqual(stat['valid'],2)
            self.assertEqual(stat['total'],5)
            self.assertAlmostEqual(stat['mean'],0.4)
            self.assertEqual(stat['missing'],3)
            self.assertEqual(len({r['sample_id'] for r in rows}),5)

    def test_visual_summary_preserves_coverage_without_overall(self):
        import pandas as pd
        from visualize_results import compute_summary
        summary=compute_summary([dict(strategy_full='baseline',label='base',color='gray',kind='stage',order=0,
            df=pd.DataFrame({'answer_relevancy':[0.8,math.nan]}))])
        self.assertNotIn('overall',summary.columns)
        self.assertEqual(summary.iloc[0]['answer_relevancy_valid'],1)
        self.assertEqual(summary.iloc[0]['total'],2)
        self.assertEqual(summary.iloc[0]['faithfulness_valid'],0)

    def test_csv_replay_parses_contexts_without_eval(self):
        from diagnostics import read_replay_csv
        import csv
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'in.csv'
            with p.open('w') as f:
                w=csv.DictWriter(f,fieldnames=['user_input','response','reference','retrieved_contexts'])
                w.writeheader(); w.writerow(dict(user_input='q',response='a',reference='r',retrieved_contexts="['a\\nb', 'c']"))
            self.assertEqual(read_replay_csv(p)[0]['retrieved_contexts'],['a\nb','c'])

if __name__=='__main__': unittest.main()
