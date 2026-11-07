"""Fresh-kernel offline replay acceptance in an isolated temporary directory with network, fitting and generation blocked."""
from pathlib import Path
import json
import os
import shutil
import sys
import tempfile
import time

import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager

PROJECT = Path(__file__).resolve().parents[1]


def main():
    started = time.monotonic()
    temp = Path(tempfile.mkdtemp(prefix="trick-of-the-tail-portable-"))
    for name in ("trick_of_the_tail.ipynb", "latest-run.json", "frozen-study.json", "pilot-admission.json"):
        shutil.copy2(PROJECT / name, temp / name)
    for folder in ("data", "sources", "calibration"):
        shutil.copytree(PROJECT / folder, temp / folder)

    run_rel = json.loads((temp / "latest-run.json").read_text())["path"]
    shutil.copytree(PROJECT / run_rel, temp / run_rel)
    if (PROJECT / "graphs").exists() and any((PROJECT / "graphs").iterdir()):
        shutil.copytree(PROJECT / "graphs", temp / "graphs")

    kernel_dir = temp / "kernels" / "trick_of_the_tail"
    kernel_dir.mkdir(parents=True)
    (kernel_dir / "kernel.json").write_text(
        json.dumps(
            dict(
                argv=[sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
                display_name="Python 3 (trick-of-the-tail validation)",
                language="python",
            )
        )
    )
    manager = KernelManager(
        kernel_name="trick_of_the_tail",
        kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(temp / "kernels")]),
    )

    nb = nbformat.read(temp / "trick_of_the_tail.ipynb", as_version=4)
    originals = {}
    for idx, cell in enumerate(nb.cells):
        if cell.cell_type != "code":
            continue
        if cell.source.startswith("from datetime import datetime"):
            originals[idx] = cell.source
            cell.source += (
                "\nimport socket\n"
                "def _network_blocked(*args, **kwargs):\n"
                "    raise RuntimeError('External network access blocked during replay acceptance')\n"
                "socket.socket.connect = _network_blocked\n"
                "socket.socket.connect_ex = _network_blocked\n"
                "def _arch_fit_blocked(*args, **kwargs):\n"
                "    raise RuntimeError('Live arch SLSQP fitting blocked during replay acceptance')\n"
                "ConstantMean.fit = _arch_fit_blocked\n"
            )
        if cell.source.startswith('frozen = json.loads(Path("frozen-study.json")'):
            originals[idx] = cell.source
            cell.source = (
                "def _generation_blocked(*args, **kwargs):\n"
                "    raise RuntimeError('Live calibration/fitting/study generation blocked during replay acceptance')\n"
                "run_study = _generation_blocked\n"
                "calibrate_pivotal = _generation_blocked\n"
                "fit_volatility_filter = _generation_blocked\n"
                + cell.source
            )

    os.environ["TAILS_MODE"] = "replay"
    os.environ["MPLCONFIGDIR"] = str(temp / "mpl-cache")
    client = NotebookClient(nb, km=manager, timeout=300, resources={"metadata": {"path": str(temp)}})
    try:
        client.execute()
    finally:
        if manager.has_kernel:
            manager.shutdown_kernel(now=True)

    for idx, src in originals.items():
        nb.cells[idx].source = src

    errors = [o for c in nb.cells if c.cell_type == "code" for o in c.outputs if o.output_type == "error"]
    if errors:
        raise RuntimeError(f"Notebook execution produced errors: {errors[0]}")
    images = [o for c in nb.cells if c.cell_type == "code" for o in c.outputs if "image/png" in o.get("data", {})]
    if len(images) != 13:
        raise AssertionError(f"Expected 13 inline PNG outputs, found {len(images)}")

    nbformat.write(nb, PROJECT / "trick_of_the_tail.ipynb")
    for pattern in ("*.csv", "asset-manifest.json", "checks.json", "primary-comparison.json"):
        for file_path in temp.glob(pattern):
            shutil.copy2(file_path, PROJECT / file_path.name)

    if (PROJECT / "graphs").exists() and any((PROJECT / "graphs").iterdir()):
        archive = PROJECT / "render-archive" / str(time.time_ns())
        shutil.copytree(PROJECT / "graphs", archive)
    shutil.copytree(temp / "graphs", PROJECT / "graphs", dirs_exist_ok=True)

    report = dict(
        status="passed",
        fresh_kernel=True,
        mode="replay",
        isolated_directory=str(temp),
        python=sys.executable,
        network_blocked=True,
        arch_fitting_blocked=True,
        calibration_and_study_generation_blocked=True,
        estimators_and_numpy_filter_recursions_verified=True,
        tolerances=dict(interior_rtol=1e-9, interior_atol=1e-11, filter_recursion_rtol=1e-10, filter_recursion_atol=1e-10),
        source_run=run_rel,
        inline_png_assets=len(images),
        executed_code_cells=sum(c.cell_type == "code" for c in nb.cells),
        execution_seconds=time.monotonic() - started,
        statement="Fresh-kernel acceptance recomputes interior estimators, NumPy filter recursions, Holm p-values, quantile losses, coverage tests and block bootstraps from saved evidence without network, optimizer fitting or synthetic study generation.",
    )
    (PROJECT / "fresh-kernel-validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

