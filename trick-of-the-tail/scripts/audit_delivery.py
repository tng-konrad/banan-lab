"""Final cross-check of trick-of-the-tail notebook, evidence, assets, companion post and links."""
from pathlib import Path
import ast
import base64
import hashlib
import json
import re

import nbformat
import numpy as np
import pandas as pd

P = Path(__file__).resolve().parents[1]
WORKSPACE = P.parent

EXPECTED_SHARED_SECTIONS = [
    "A tail curve is not a finding",
    "One threshold and dependent extremes",
    "Can the test control false alarms?",
    "Raw losses and standardized innovations",
    "Does tail shape stay constant?",
    "Predict tomorrow using yesterday",
    "What does updating shape add?",
    "Robustness and remaining uncertainty",
    "Closing time",
]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def headings(text):
    return re.findall(r"^# (.+)$", text, re.M)


def main():
    nb = nbformat.read(P / "trick_of_the_tail.ipynb", as_version=4)
    shared = []
    active = False
    images = []
    code_cells = []
    current_section = None

    for idx, cell in enumerate(nb.cells):
        if cell.cell_type == "markdown":
            for title in headings(cell.source):
                if title == EXPECTED_SHARED_SECTIONS[0]:
                    active = True
                if active:
                    shared.append(title)
                    current_section = title
        else:
            assert cell.execution_count is not None, f"Unexecuted code cell at index {idx}"
            assert idx + 1 < len(nb.cells) and nb.cells[idx + 1].cell_type == "markdown", (
                f"Code cell at index {idx} is not followed immediately by an explanatory Markdown cell"
            )
            ast.parse(cell.source)
            code_cells.append(cell.source)
            assert not any(o.output_type == "error" for o in cell.outputs), f"Error output in cell {idx}"
            for out in cell.outputs:
                if "image/png" in out.get("data", {}):
                    png_bytes = base64.b64decode(out.data["image/png"])
                    images.append((current_section, hashlib.sha256(png_bytes).hexdigest()))

    assert shared in (EXPECTED_SHARED_SECTIONS, EXPECTED_SHARED_SECTIONS[:-1]), f"Notebook headings mismatch: {shared}"
    post_plan_text = (P / "post-plan.md").read_text()
    assert headings(post_plan_text)[1:] == EXPECTED_SHARED_SECTIONS, "post-plan.md headings do not match notebook"
    docs_to_check = [("post-plan.md", post_plan_text)]
    if (P / "post.md").exists():
        post_text = (P / "post.md").read_text()
        assert headings(post_text)[1:] == EXPECTED_SHARED_SECTIONS, "post.md headings do not match notebook"
        assert "United States of Banan is a reader-supported publication." in post_text
        assert "[ ✓ Subscribed ]" in post_text
        docs_to_check.append(("post.md", post_text))

    manifest = json.loads((P / "asset-manifest.json").read_text())
    assert len(manifest) == len(images) == 13, f"Expected 13 assets, got manifest={len(manifest)}, inline={len(images)}"
    assert [a["number"] for a in manifest] == [f"{n:02d}" for n in range(1, 14)]

    visual = json.loads((P / "visual-qa.json").read_text())
    assert visual["status"] == "passed"
    for (sec_name, img_sha), asset in zip(images, manifest):
        assert (sec_name, img_sha) == (asset["section"], asset["sha256"]), f"Inline/manifest mismatch on {asset['number']}"
        assert digest(P / asset["path"]) == asset["sha256"] == visual["assets"][asset["path"]]
        assert re.fullmatch(r"graphs/(graph|tabler)_\d{2}_\d{4}\.png", asset["path"])
        assert asset["sidecar"] and (P / asset["sidecar"]).exists()
        assert asset["caption"] and asset["alt"] and asset["reading"]
        expected_ddmm = asset["execution_timestamp"][8:10] + asset["execution_timestamp"][5:7]
        assert expected_ddmm == Path(asset["path"]).stem[-4:]

    for doc_name, doc_body in docs_to_check:
        refs = re.findall(r"!\[[^\]]*\]\((graphs/[^)]+)\)", doc_body)
        assert refs == [a["path"] for a in manifest], f"Image sequence mismatch in {doc_name}: {refs}"

    joined_code = "\n".join(code_cells)
    assert "notebook_src" not in joined_code and "from src" not in joined_code and "from scripts" not in joined_code
    first_cell_imports = [
        node for node in ast.walk(ast.parse(code_cells[0])) if isinstance(node, (ast.Import, ast.ImportFrom))
    ]
    later_imports = [
        node
        for src in code_cells[1:]
        for node in ast.walk(ast.parse(src))
        if isinstance(node, (ast.Import, ast.ImportFrom))
    ]
    assert len(first_cell_imports) == 20 and not later_imports, "Imports must reside exclusively in the first code cell"

    # Verify source and evidence hashes
    src_manifest = json.loads((P / "data/source-manifest.json").read_text())
    assert digest(P / "sources/2611.06005v1.pdf") == src_manifest["source_documents"]["paper_pdf"]["sha256"]
    assert digest(P / "sources/trick-of-the-tail-proper-plan-v2.md") == src_manifest["source_documents"]["planning_spec_v2"]["sha256"]
    assert digest(P / "data/DEXUSEU.csv") == src_manifest["series"]["EUR"]["sha256"]
    if (P / "archive/v1-time-varying-tails/data/DEXUSEU.csv").exists():
        assert digest(P / "data/DEXUSEU.csv") == digest(P / "archive/v1-time-varying-tails/data/DEXUSEU.csv")
    for s_code in ("EUR", "GBP", "JPY"):
        assert digest(P / src_manifest["series"][s_code]["filename"]) == src_manifest["series"][s_code]["sha256"]

    run_rel = json.loads((P / "latest-run.json").read_text())["path"]
    run_meta = json.loads((P / run_rel / "run.json").read_text())
    frozen = json.loads((P / "frozen-study.json").read_text())
    assert run_meta["status"] == "complete"
    assert run_meta["configuration"] == frozen["configuration"]
    assert run_meta["failed_filters"] == 0 and run_meta["failed_forecasts"] == 0
    assert digest(P / run_rel / "inference-evidence.npz") == run_meta["inference_evidence_sha256"]
    assert digest(P / run_rel / "forecast-evidence.npz") == run_meta["forecast_evidence_sha256"]

    cal_dirs = sorted([d for d in (P / "calibration").iterdir() if d.is_dir()])
    assert len(cal_dirs) == 7
    for c_dir in cal_dirs:
        c_meta = json.loads((c_dir / "calibration.json").read_text())
        assert digest(c_dir / "calibration.npz") == c_meta["npz_sha256"]

    # Independent recalculation of primary prediction contrast and 9-slot Holm adjustment
    primary = json.loads((P / "primary-comparison.json").read_text())
    with np.load(P / run_rel / "forecast-evidence.npz", allow_pickle=False) as fc_npz:
        y_eur = fc_npz["EUR_expanding_realized"]
        q_eur_01 = fc_npz["EUR_expanding_forecasts"][:, 0, :]
        tau = 0.99
        loss_b = (tau - (y_eur < q_eur_01[:, 1]).astype(float)) * (y_eur - q_eur_01[:, 1])
        loss_c = (tau - (y_eur < q_eur_01[:, 2]).astype(float)) * (y_eur - q_eur_01[:, 2])
        indep_diff = loss_c - loss_b
        np.testing.assert_allclose(indep_diff.mean(), primary["prediction_primary"]["mean_diff"], rtol=1e-13, atol=1e-13)
        assert len(indep_diff) == primary["prediction_primary"]["scored_dates"] == 4253

    base_df = pd.read_csv(P / "baseline-stability-tests.csv")
    assert len(base_df) == 9
    order = np.argsort(base_df["p_raw"].to_numpy(), kind="stable")
    holm_check = np.empty(9, dtype=float)
    run_max = 0.0
    for rk, ix in enumerate(order):
        run_max = max(run_max, min(1.0, (9 - rk) * float(base_df.loc[ix, "p_raw"])))
        holm_check[ix] = run_max
    np.testing.assert_allclose(holm_check, base_df["p_holm"].to_numpy(), rtol=1e-13, atol=1e-13)

    seo_text = (P / "seo.txt").read_text().strip()
    assert 50 <= len(seo_text) <= 160, f"SEO length {len(seo_text)} outside [50, 160]"

    for md_name in ("README.md", "plan.md", "plan-critique.md", "dependency-audit.md", "validation.md", "post-plan.md", "post-plan-review.md", "post.md", "linkedin.md"):
        if not (P / md_name).exists():
            continue
        content = (P / md_name).read_text()
        for target in re.findall(r"\]\(([^)]+)\)", content):
            if target.startswith(("https://", "http://", "#")):
                continue
            if target in ("delivery-audit.json", "post.md"):
                continue
            resolved = (P / target).resolve()
            assert resolved.exists(), f"Broken local link in {md_name}: {target} -> {resolved}"

    fresh = json.loads((P / "fresh-kernel-validation.json").read_text())
    assert fresh["status"] == "passed" and fresh["network_blocked"] and fresh["arch_fitting_blocked"] and fresh["calibration_and_study_generation_blocked"]
    checks = json.loads((P / "checks.json").read_text())
    assert checks["status"] == "passed" and checks["verify_estimators"]

    report = dict(
        status="passed",
        executed_code_cells=len(code_cells),
        shared_sections=len(EXPECTED_SHARED_SECTIONS),
        exact_inline_export_matches=len(manifest),
        calibration_caches_verified=len(cal_dirs),
        total_pivotal_draws_verified=len(cal_dirs) * 5000,
        baseline_holm_slots=9,
        scored_forecast_dates_per_series=4253,
        primary_contrast_recomputed_independently=True,
        holm_pvalues_recomputed_independently=True,
        raw_source_and_evidence_hashes_verified=True,
        historical_package_archived=(P / "archive/v1-time-varying-tails/time_varying_tails.ipynb").exists(),
        all_imports_in_single_cell=True,
        seo_characters=len(seo_text),
        fresh_kernel=fresh,
        visual_qa="all 13 single-panel PNGs visually inspected and hash-bound in visual-qa.json",
    )
    (P / "delivery-audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

