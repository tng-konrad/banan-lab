"""Meaningful data, decoding, objective and metric fixtures for the v4 notebook."""
from pathlib import Path
import json
import unittest
from unittest.mock import patch
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
NB = json.loads((ROOT / "stair_local_demo.ipynb").read_text())
NS = {}
for cell in NB["cells"]:
    if cell["cell_type"] != "code":
        continue
    source = "".join(cell["source"])
    if source.startswith("if CFG.MODE == 'live':"):
        break
    exec(compile(source, "<standalone v4 notebook>", "exec"), NS)


class V4Tests(unittest.TestCase):
    def test_same_bytes_preserve_v3_failure_and_v4_warning(self):
        old = NS["audit_release"]("source_leaf_v3")
        new = NS["audit_release"]("released_document_anchor_v4")
        self.assertEqual(old["status"], "stopped_at_data_gate")
        self.assertEqual(new["status"], "data_gate_passed_with_boundary_warnings")
        self.assertEqual(new["confirmed_multi_leaf_documents"], 9)
        self.assertEqual(new["historical_source_leaf_gate"]["status"], "failed")
        self.assertEqual(new["document_audit"], old["document_audit"])
        self.assertEqual(new["crosswalk"], old["crosswalk"])

    def test_complete_source_identity_and_parent_paths(self):
        audit = NS["audit_release"]()
        hierarchy = NS["build_anchor_hierarchy"](audit)
        self.assertEqual(len(hierarchy["target_ids"]), 86)
        self.assertEqual(len(set(hierarchy["target_ids"])), 86)
        targets = [row for row in hierarchy["nodes"] if row["target"]]
        self.assertEqual({row["label"] for row in targets}, {row["canonical_title"] for row in audit["crosswalk"]})
        four = next(row for row in targets if row["label"] == "6.4 Issuing Stock for Non-Cash Assets")
        five = next(row for row in targets if row["label"] == "6.5 Treasury Stock")
        self.assertNotEqual(four["source_id"], five["source_id"])
        self.assertEqual(four["parent_id"], "chapter-6")
        self.assertEqual(five["parent_id"], "chapter-6")

    def test_trie_common_prefix_eos_and_finished_slots(self):
        trie = NS["TargetTrie"]([[2, 1], [2, 3, 1]])
        self.assertEqual(trie.allowed([]), [2])
        self.assertEqual(trie.allowed([2]), [1, 3])
        self.assertEqual(trie.allowed([2, 3]), [1])
        self.assertEqual(trie.allowed([2, 1]), [1])
        self.assertEqual(trie.allowed([2, 1, 0]), [1])
        with self.assertRaises(ValueError):
            trie.allowed([9])
        with self.assertRaises(ValueError):
            trie.allowed([2, 1, 7])

    def test_mask_keeps_full_vocabulary_log_probabilities(self):
        trie = NS["TargetTrie"]([[2, 1], [3, 1]])
        mask = NS["TrieMask"](trie)
        probabilities = torch.log_softmax(torch.tensor([[0., 1., 2., 3.]]), dim=1)
        masked = mask(torch.tensor([[0]]), probabilities)
        self.assertTrue(torch.equal(masked[0, 2:], probabilities[0, 2:]))
        self.assertLess(float(masked.exp().sum()), 1.)
        self.assertTrue(torch.isneginf(masked[0, 0]))

    def test_equal_example_loss_does_not_weight_long_titles_more(self):
        logits = torch.tensor([[[2., 0.], [0., 0.], [0., 0.]],
                               [[0., 2.], [0., 2.], [0., 2.]]])
        labels = torch.tensor([[0, -100, -100], [0, 0, 0]])
        losses = NS["per_example_loss"](logits, labels)
        expected = torch.tensor([np.log1p(np.exp(-2.)), np.log1p(np.exp(2.))], dtype=losses.dtype)
        self.assertTrue(torch.allclose(losses, expected, atol=1e-6))
        self.assertAlmostEqual(float(losses.mean()), float(expected.mean()), places=6)

    def test_short_accumulation_group_matches_full_example_mean(self):
        class Toy(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(1, 2, bias=False)
                torch.nn.init.zeros_(self.linear.weight)
            def forward(self, input_ids, labels, use_cache=False):
                return type("Output", (), {"logits": self.linear(input_ids.float().unsqueeze(-1) / 10)})()
        rows = [{"x": x, "y": x % 2} for x in range(1, 6)]
        def collate(_, fmt, subset, hierarchy, device):
            return {"input_ids": torch.tensor([[r["x"]] for r in subset])}, torch.tensor([[r["y"]] for r in subset])
        full, split = Toy(), Toy()
        with patch.dict(NS, {"collate_training": collate}):
            NS["train_update"](full, None, torch.optim.SGD(full.parameters(), lr=.1), "D", rows, {}, "cpu", 5)
            NS["train_update"](split, None, torch.optim.SGD(split.parameters(), lr=.1), "D", rows, {}, "cpu", 2)
        self.assertTrue(torch.allclose(full.linear.weight, split.linear.weight, atol=1e-7))

    def test_invalid_and_duplicate_ranks_do_not_compress(self):
        record = [{"section_id": None, "validity": "nonexistent_title"},
                  {"section_id": "gold", "validity": "valid"},
                  {"section_id": "gold", "validity": "valid"}]
        values = NS["retrieval_values"](record, "gold")
        self.assertEqual(values["r1"], 0)
        self.assertEqual(values["r3"], 1)
        self.assertEqual(values["invalid1"], 1)
        self.assertAlmostEqual(values["ndcg3"], 1 / np.log2(3))

    def test_prompts_ignore_answer_and_gold_metadata(self):
        hierarchy = {"flat_text": "all candidate labels", "hierarchical_text": "verified ancestors and labels"}
        query = {"query": "Which entry?", "answer": "SECRET", "gold_title": "GOLD"}
        for fmt in "CDEF":
            text = NS["prompt_text"](fmt, query["query"], hierarchy)
            self.assertNotIn("SECRET", text)
            self.assertNotIn("GOLD", text)
        self.assertNotIn("all candidate labels", NS["prompt_text"]("D", query["query"], hierarchy))


if __name__ == "__main__":
    unittest.main()
