import importlib.util
from pathlib import Path
import unittest
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("improvement", ROOT / "notebook_src/stair_improvement.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
torch = m.torch


class MarkovFixture(torch.nn.Module):
    """Explicit transition probabilities; synthetic unit fixture, never experiment evidence."""
    def __init__(self):
        super().__init__()
        p = torch.full((7, 7), .01)
        p[0] = torch.tensor([.01, .01, .55, .20, .21, .01, .01])
        p[2] = torch.tensor([.01, .50, .01, .44, .01, .02, .01])
        p[3] = torch.tensor([.01, .93, .01, .01, .01, .02, .01])
        p[4] = torch.tensor([.01, .94, .01, .01, .01, .01, .01])
        self.register_buffer("logp", p.log())
        self.encoder_calls = 0

    def get_encoder(self):
        def encode(input_ids, attention_mask, return_dict):
            self.encoder_calls += 1
            return SimpleNamespace(last_hidden_state=input_ids.float().unsqueeze(-1))
        return encode

    def _shift_right(self, labels):
        decoder = torch.zeros_like(labels)
        decoder[:, 1:] = labels[:, :-1].clamp_min(0)
        return decoder

    def forward(self, encoder_outputs, attention_mask, decoder_input_ids, decoder_attention_mask,
                use_cache, return_dict):
        return SimpleNamespace(logits=self.logp[decoder_input_ids])


class ImprovementTests(unittest.TestCase):
    def test_recovery_clock_retains_prior_charge_without_charging_wall_clock_jump(self):
        guard=object.__new__(m.CheapGuard)
        guard.active_clock=True
        guard.prior_charge_seconds=2907.6
        guard.started_monotonic=1000.
        guard.phase_started_monotonic=900.
        guard.started_wall=100.
        guard.phase_started=100.
        with patch.object(m.time,"monotonic",return_value=1060.), patch.object(m.time,"time",return_value=50000.):
            self.assertAlmostEqual(guard.elapsed(),2967.6)
            self.assertEqual(guard.phase_elapsed(),160.)
            guard.active_clock=False
            self.assertEqual(guard.elapsed(),49900.)
            self.assertEqual(guard.phase_elapsed(),49900.)

    def test_stream_matches_historical_order_when_pool_divisible_by_eight(self):
        rows = list(range(256))
        stream = m.training_groups(rows)
        actual = [item for _ in range(128) for item in next(stream)]
        expected = [int(x) for epoch in range(1, 5)
                    for x in m.np.random.default_rng(42002 + epoch).permutation(256)]
        self.assertEqual(actual, expected)

    def test_stream_carries_remainder_without_dropping_or_changing_update_size(self):
        stream = m.training_groups(list(range(19)))
        groups = [next(stream) for _ in range(10)]
        self.assertTrue(all(len(group) == 8 for group in groups))
        expected = [int(x) for epoch in range(1, 6)
                    for x in m.np.random.default_rng(42002 + epoch).permutation(19)][:80]
        self.assertEqual([x for group in groups for x in group], expected)

    def test_exact_scoring_matches_hand_calculation_with_eos_and_prefix(self):
        model = MarkovFixture()
        paths = [[2, 1], [2, 3, 1], [4, 1]]
        scores, _ = m.exact_scores(model, torch.tensor([[5, 6]]), torch.ones(1, 2, dtype=torch.long), paths, 2)
        expected = [(.55 * .50) ** .5, (.55 * .44 * .93) ** (1 / 3), (.21 * .94) ** .5]
        self.assertTrue(m.np.allclose([s['score'] for s in scores], m.np.log(expected), atol=1e-6))
        self.assertEqual(sorted(range(3), key=lambda i: -scores[i]['score']), [1, 0, 2])
        self.assertEqual(model.encoder_calls, 1)

    def test_exact_scorer_matches_independent_t5_forward_and_batching(self):
        from transformers import T5Config, T5ForConditionalGeneration
        torch.manual_seed(19)
        model = T5ForConditionalGeneration(T5Config(vocab_size=13, d_model=16, d_ff=32, num_layers=1,
            num_decoder_layers=1, num_heads=2, dropout_rate=0., decoder_start_token_id=0, pad_token_id=0, eos_token_id=1)).eval()
        paths = [[2, 1], [2, 3, 1], [4, 5, 6, 1]]
        source = torch.tensor([[7, 8, 1]])
        mask = torch.ones_like(source)
        scores, _ = m.exact_scores(model, source, mask, paths, 3)
        scalar, _ = m.exact_scores(model, source, mask, paths, 1)
        self.assertTrue(m.np.allclose([x['score'] for x in scores], [x['score'] for x in scalar], atol=1e-5))
        with torch.no_grad():
            for path, row in zip(paths, scores):
                labels = torch.tensor([path])
                logits = model(input_ids=source, attention_mask=mask, labels=labels).logits
                independent = logits.log_softmax(-1).gather(-1, labels.unsqueeze(-1)).mean().item()
                self.assertAlmostEqual(independent, row['score'], places=5)

    def test_shadow_stopping_does_not_peek_at_later_recovery(self):
        scores = [.5, .5, .4, .4, .4, .6, .7]
        history = [{'update': (i + 1) * 32, 'tune_r1': score, 'checkpoint': str(i)} for i, score in enumerate(scores)]
        result = m.shadow_selection(history)
        self.assertEqual(result['would_stop_at_update'], 160)
        self.assertEqual(result['selected_update'], 32)

    def test_paired_clusters_preserve_pairing_and_detect_incomplete_comparison(self):
        corpus = [{'title':'A'}, {'title':'B'}]
        queries = [{'query_id':str(i),'gold_title':title} for i,title in enumerate(['A','A','B'])]
        left = [{'method':'F','mode':'constrained','query_id':str(i),'hypotheses':[
                 {'section_id':'finance:000','validity':'valid'}]} for i in range(3)]
        right = [{'method':'F','mode':'exact','query_id':str(i),'hypotheses':[
                  {'section_id':'finance:000' if i<2 else 'finance:001','validity':'valid'}]} for i in range(3)]
        result = m.paired_summary(left,right,queries,corpus)
        self.assertEqual(result['gains'],1)
        self.assertEqual(result['losses'],0)
        self.assertAlmostEqual(result['difference_pp'],100/3)
        with self.assertRaises(ValueError):
            m.paired_summary(left,right[:-1],queries,corpus)


if __name__ == '__main__':
    unittest.main()
