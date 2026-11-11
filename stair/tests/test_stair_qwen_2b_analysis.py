"""Synthetic raw-record fixtures, kept outside all experiment runs; no model calls."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('pilot_offline',ROOT/'scripts/analyze_stair_qwen_2b_pilot.py')
a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)


class OfflinePilotTests(unittest.TestCase):
    def fixture(self,folder,qwen_correct):
        source=ROOT/'runs/stair/improvement-pilots/cheap-recovery-20261107T104810314921Z'
        data=json.loads((source/'inputs.json').read_text())
        def write(path,value):
            path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value)+'\n')
        write(folder/'inputs.json',data)
        write(folder/'screen_semantic_audit.json',json.loads((source/'screen_semantic_audit.json').read_text()))
        write(folder/'quality/status.json',{'status':'complete'})
        write(folder/'flan_baseline/result.json',{'status':'complete'})
        write(folder/'quality/process_status.json',{'elapsed_seconds':600})
        write(folder/'quality_admission.json',{'training_microbatch':4,'inference_batch':1})
        cells={'C':{'method':'C','budget':0,'selected_update':0,'training_seconds':0,'checkpoint_sha256':'base'}}
        for b in [128,384]:
            for f in 'DEF':cells[f'n935_u{b}_{f}']={'method':f,'budget':b,'selected_update':b,'training_seconds':b,
                'checkpoint_sha256':f'{f}-{b}'}
        cells['n935_u128_D']['selected_update']=32
        write(folder/'quality/cells.json',cells)
        ids={r['title']:f'finance:{i:03d}' for i,r in enumerate(data['corpus'])}
        def raw(correct,method):
            result=[]
            for query in data['screen']:
                gold=ids[query['gold_title']]
                sid=gold if correct else ('finance:000' if gold!='finance:000' else 'finance:001')
                result.append({'query_id':query['query_id'],'method':method,'hypotheses':[
                    {'section_id':sid,'validity':'valid','token_ids':[2,1],'token_logprobs':[-1.,-1.],'score':-1.}]*3,
                    'timing':{'batch_size':1,'end_to_end_seconds':.01}})
            return result
        for family in ['quality','flan_baseline']:
            for key,d in cells.items():
                values=raw(qwen_correct if family=='quality' else not qwen_correct,d['method'])
                for mode in ['constrained','unconstrained']:
                    path=folder/family/'predictions'/key/(mode+'.jsonl');path.parent.mkdir(parents=True,exist_ok=True)
                    path.write_text(''.join(json.dumps(row)+'\n' for row in values))
        lexical=[]
        for f in 'AB':lexical+=raw(False,f)
        p=folder/'flan_baseline/lexical/lexical.jsonl';p.parent.mkdir(parents=True,exist_ok=True)
        p.write_text(''.join(json.dumps(row)+'\n' for row in lexical))

    def test_perfect_upgrade_is_recomputed_and_not_called_hierarchy_benefit(self):
        with tempfile.TemporaryDirectory(prefix='stair-qwen-analysis-fixture-') as path:
            run=Path(path);self.fixture(run,True);result=a.analyze(run)
            upgrade=next(r for r in result['effects'] if r['contrast']=='backbone_upgrade_F_384' and r['right'][2]=='constrained')
            hierarchy=next(r for r in result['effects'] if r['contrast']=='F_minus_E_384' and r['right'][2]=='constrained')
            self.assertEqual(upgrade['difference_pp'],100.)
            self.assertEqual(hierarchy['difference_pp'],0.)
            self.assertEqual(len(result['shortlist']),2)
            early=result['selected_checkpoint_exposure']['qwen/n935_u128_D']
            self.assertEqual(early['distinct_queries_seen'],256)
            self.assertEqual(sum(early['example_passes_by_title'].values()),256)
            self.assertLess(early['distinct_queries_seen'],935)
            self.assertIn('Low-support diagnostic',a.report(run,result))

    def test_worse_backbone_is_not_promoted(self):
        with tempfile.TemporaryDirectory(prefix='stair-qwen-analysis-fixture-') as path:
            run=Path(path);self.fixture(run,False);result=a.analyze(run)
            self.assertEqual(result['matched_upgrade_leads'],[])
            self.assertEqual(result['shortlist'],[])


if __name__=='__main__':unittest.main()
