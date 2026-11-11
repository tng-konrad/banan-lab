"""Synthetic cross-backbone evidence outside all experiment directories."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('four_b', ROOT/'scripts/analyze_stair_qwen_4b_pilot.py')
a = importlib.util.module_from_spec(spec)
spec.loader.exec_module(a)
spec = importlib.util.spec_from_file_location('fixture', ROOT/'tests/test_stair_qwen_2b_analysis.py')
fixture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixture)


class FourBAnalysisTests(unittest.TestCase):
    def make(self,folder,better):
        two=folder/'two';four=folder/'four'
        helper=fixture.OfflinePilotTests()
        helper.fixture(two,not better);helper.fixture(four,better)
        (four/'config.json').write_text(json.dumps({'two_b_run':str(two)})+'\n')
        return four

    def test_primary_is_flat_at_full_budget_and_does_not_imply_hierarchy_gain(self):
        with tempfile.TemporaryDirectory(prefix='stair-4b-offline-') as folder:
            run=self.make(Path(folder),True);result=a.analyze(run)
            self.assertEqual(result['primary']['difference_pp'],100.)
            self.assertEqual(result['primary']['right'],['qwen4b','n935_u384_E','constrained'])
            hierarchy=next(e for e in result['effects'] if e['contrast']=='F_minus_E_384' and e['right'][2]=='constrained')
            self.assertEqual(hierarchy['difference_pp'],0.)
            self.assertEqual({r['family'] for r in result['metrics']},{'qwen4b','qwen2b','flan','lexical'})
            self.assertIn('primary comparison',a.report(run,result))

    def test_worse_four_b_is_not_promoted(self):
        with tempfile.TemporaryDirectory(prefix='stair-4b-offline-') as folder:
            run=self.make(Path(folder),False);result=a.analyze(run)
            self.assertEqual(result['primary']['difference_pp'],-100.)
            self.assertEqual(result['primary']['lead'],'not prioritized')

    def test_interruption_cost_separates_worker_time_from_waiting_and_replayed_tail(self):
        with tempfile.TemporaryDirectory(prefix='stair-4b-recovery-cost-') as folder:
            run = Path(folder)
            (run/'quality').mkdir()
            evidence = {
                'recovery_provenance.json': {'source_run': 'preserved-original',
                    'source_process_status': {'attempt_elapsed_seconds': 8, 'watchdog': 'Parent memory guard'},
                    'discarded_completed_tail': {'D': [{'seconds': 3}]},
                    'copied_updates': {'D': 320}, 'original_quality_origin_wall': 100},
                'quality/process_status.json': {'attempt_elapsed_seconds': 12, 'elapsed_seconds': 30},
                'quality/clock.json': {'started_wall': 100},
                'config.json': {'available_memory_floor_gib': 32}}
            for name, value in evidence.items():
                (run/name).write_text(json.dumps(value))
            result = a.recovery_costs(run)
            self.assertEqual(result['quality_worker_attempt_seconds'], 20)
            self.assertEqual(result['time_outside_quality_workers_seconds'], 10)
            self.assertEqual(result['discarded_update_seconds'], 3)
            self.assertEqual(result['discarded_updates_replayed'], 1)
            self.assertTrue(result['original_quality_clock_retained'])


if __name__=='__main__':unittest.main()
