"""Setup cell; copied inline into the standalone notebook by the authoring script."""
# Standard library
from collections import Counter, defaultdict
from datetime import datetime, timezone
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import random
import shutil
import subprocess
import textwrap
import time
import unicodedata
import threading
from contextlib import contextmanager

# Keep Matplotlib's cache writable in isolated notebook execution.
os.environ.setdefault("MPLCONFIGDIR", str(Path(os.environ.get("STAIR_ROOT", ".")).resolve() / "data/stair/mpl-cache"))
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_OFFLINE"] = "0" if os.environ.get("STAIR_MODE") == "prepare" else "1"

# Third-party packages: requests/preparation, analysis, graphics and display.
import httpx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
import torch
import torch.nn.functional as torch_F
from rank_bm25 import BM25Okapi
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, GenerationConfig, LogitsProcessor, LogitsProcessorList
from huggingface_hub import snapshot_download
from IPython.display import Image, Markdown, display


class CFG:
    MODE = os.environ.get("STAIR_MODE", "report")
    root = Path(os.environ.get("STAIR_ROOT", ".")).resolve()
    data_folder = root / "data/stair"
    run_folder = root / "runs/stair"
    graph_folder = root / "graphs/stair"
    dataset = "meghanadh/SearchTome"
    dataset_revision = "48d3b3545950b3857dc3657fb22e4b79b8ed8df5"
    dataset_files = ("finance/corpus.jsonl", "finance/train.jsonl", "finance/dev.jsonl",
                     "finance/test.jsonl", "README.md", "LICENSE", "stats.json")
    source_pdf_url = "https://open.umn.edu/opentextbooks/formats/1101"
    source_pdf_sha256 = "ddc74629d85972a05bc362c0e3f3a23e678b4869c8b97a512aba00bf1a220174"
    source_toc_pdf_pages = (5, 6, 7, 8, 9)  # One-based physical PDF pages.
    source_body_first_pdf_page = 11
    seed = 42
    encoder_cap = 4096
    decoder_cap = 96
    preferred_profile = {"train": 256, "dev": 48, "test": 120, "epochs": 4}
    fallback_profile = {"train": 128, "dev": 32, "test": 60, "epochs": 3}
    measured_budget_minutes = 180
    pilot_budget_minutes = 30
    model = "google/flan-t5-small"
    model_revision = "0fc9ddf78a1e988dac52e2dac162b0ede4fd74ab"
    preferred_backend = "mps"
    model_dtype = "float32"
    target_contract = "released_document_anchor_v4"
    effective_batch = 8
    batch_accumulation_candidates = ((4, 2), (2, 4), (1, 8))
    inference_batch_candidates = (1, 2, 4)
    model_folder = data_folder / "model"
    learning_rate = 1e-4
    bootstrap_seed = 42001
    order_seed = 42002
    bootstrap_replicates = 10000
    rss_limit_gib = 48
    available_floor_gib = 32
    swap_growth_limit_gib = 1
    resume_run_id = os.environ.get("STAIR_RUN_ID")
    img_dim1, img_dim2 = 9.5, 6.5
    example_figure_height = 8.0
    export_dpi = 200
    asset_suffix = "0311"


if CFG.MODE not in {"prepare", "report", "live"}:
    raise ValueError("STAIR_MODE must be prepare, report or live")
plt.style.use("seaborn-v0_8")
plt.rcParams.update({"figure.figsize": (CFG.img_dim1, CFG.img_dim2), "font.size": 11})
np.random.seed(CFG.seed)
print(f"Mode: {CFG.MODE}; inputs: {CFG.data_folder}")
