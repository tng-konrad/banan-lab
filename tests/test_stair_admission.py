"""Regression checks for the archived v3 source-leaf admission failure."""
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
NB = json.loads((ROOT / "runs/stair/history/v3-source-leaf-contract/stair_local_demo.ipynb").read_text())
NS = {}
for cell in [cell for cell in NB["cells"] if cell["cell_type"] == "code"][:4]:
    exec(compile("".join(cell["source"]), "<delivered notebook utils>", "exec"), NS)


class AdmissionTests(unittest.TestCase):
    def test_title_canonicalization_keeps_case_distinctions(self):
        self.assertEqual(NS["canonical"]("  Café\n  title "), "Café title")
        self.assertEqual(NS["canonical"]("Cafe\u0301 title"), "Café title")
        self.assertNotEqual(NS["canonical"]("Title"), NS["canonical"]("title"))

    def test_toc_wrapping_single_dot_and_repeated_number(self):
        text = """6.4 Non-Cash Assets . . . 239
6.4 Treasury Stock . . . 241
7.2 Cash Flows . . . 256
7.2.1 A long heading
that wraps . 256
"""
        with patch.object(NS["CFG"], "source_toc_pdf_pages", (1,)):
            rows = NS["source_toc"](text)
        self.assertEqual(len(rows), 4)
        self.assertEqual(len({row["source_id"] for row in rows}), 4)
        self.assertEqual(rows[-1]["title"], "A long heading that wraps")
        self.assertFalse(rows[2]["leaf"])
        self.assertTrue(rows[0]["leaf"])
        self.assertTrue(rows[1]["leaf"])

    def test_unparsed_toc_entry_blocks_instead_of_disappearing(self):
        with patch.object(NS["CFG"], "source_toc_pdf_pages", (1,)):
            with self.assertRaisesRegex(ValueError, "Unparsed"):
                NS["source_toc"]("1.1 Heading without a page\n1.2 Another heading . . . 2")

    def test_complete_source_headings_detect_merge_but_not_decimal_amounts(self):
        leaves = [{"source_id": "a", "code": "1.2", "title": "Net Income—A Critical Amount",
                   "printed_page": 3, "toc_pdf_page": 5},
                  {"source_id": "b", "code": "1.3.1", "title": "The Journal",
                   "printed_page": 3, "toc_pdf_page": 5}]
        body = "1.2 NET INCOME—A CRITICAL\nAMOUNT\nSome source text.\n1.3.1 The Journal\nTransactions."
        result = NS["heading_matches"](body, leaves)
        self.assertEqual([row["source_id"] for row in result], ["a", "b"])
        self.assertEqual(NS["heading_matches"]("1.2 99.00\n1.3.1 A different title\nThe Journal", leaves), [])

    def test_pinned_real_release_exposes_required_stop(self):
        with patch.object(NS["CFG"], "data_folder", ROOT / "data/stair"):
            result = NS["audit_release"]()
        self.assertEqual(result["status"], "stopped_at_data_gate")
        self.assertEqual(result["fatal_reasons"], ["released_documents_contain_multiple_source_leaves"])
        self.assertEqual(result["corpus_documents"], 86)
        self.assertEqual(result["source_leaf_count"], 99)
        self.assertEqual(result["confirmed_multi_leaf_documents"], 9)
        self.assertEqual(result["numbering_discrepancies"], 5)
        self.assertEqual(result["mapping_problems"], [])
        self.assertEqual(result["missing_gold_titles"], [])
        self.assertEqual(result["test_predictions"], 0)

    def test_missing_cached_inputs_cannot_launch_preparation(self):
        with patch.object(NS["CFG"], "data_folder", ROOT / "data/stair/not-present"):
            with patch.object(NS["httpx"], "Client", side_effect=AssertionError("network forbidden")):
                with self.assertRaisesRegex(FileNotFoundError, "Cached STAIR evidence"):
                    NS["validate_inputs"]()

    def test_downloads_are_explicitly_gated(self):
        with patch.object(NS["CFG"], "MODE", "report"):
            with self.assertRaisesRegex(RuntimeError, "MODE='prepare'"):
                NS["prepare_sources"]()

    def test_changed_cache_fails_hash_validation(self):
        with tempfile.TemporaryDirectory(prefix="stair-cache-test-") as directory:
            target = Path(directory)
            shutil.copytree(ROOT / "data/stair/raw", target / "raw")
            shutil.copy2(ROOT / "data/stair/source_manifest.json", target / "source_manifest.json")
            with (target / "raw/finance/corpus.jsonl").open("ab") as handle:
                handle.write(b"\n")
            with patch.object(NS["CFG"], "data_folder", target):
                with self.assertRaisesRegex(ValueError, "Source identity/hash mismatch"):
                    NS["validate_inputs"]()

    def test_live_saves_gate_failure_and_raises_before_model_work(self):
        with tempfile.TemporaryDirectory(prefix="stair-live-test-") as directory:
            root = Path(directory)
            with patch.multiple(NS["CFG"], root=root, run_folder=root / "runs/stair",
                                graph_folder=root / "graphs/stair", data_folder=ROOT / "data/stair", MODE="live"):
                with patch.object(NS["httpx"], "Client", side_effect=AssertionError("network forbidden")):
                    result = NS["run_admission"]()
                    with self.assertRaisesRegex(RuntimeError, "No model work was launched"):
                        NS["finish_report"](result)
            status = json.loads((result["run"] / "status.json").read_text())
            self.assertEqual(status["status"], "stopped_at_data_gate")
            self.assertFalse(status["model_downloaded"])
            self.assertFalse(status["training_started"])
            self.assertEqual(status["test_predictions"], 0)


if __name__ == "__main__":
    unittest.main()
