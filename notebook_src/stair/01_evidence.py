def canonical(text):
    """The approved title normalization: NFC and whitespace only."""
    return " ".join(unicodedata.normalize("NFC", text).split())


def source_match_key(text):
    """Case-insensitive source-heading audit only, never a prediction evaluator."""
    return canonical(text).casefold()


def digest_bytes(payload):
    return hashlib.sha256(payload).hexdigest()


def json_read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def json_write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def package_inventory():
    result = {"python": platform.python_version(), "architecture": platform.machine(),
              "platform": platform.platform()}
    for name in ("numpy", "pandas", "matplotlib", "httpx", "nbformat", "nbclient", "torch", "transformers"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def prepare_sources():
    """Explicit preparation only; model downloads/inference are never launched here."""
    if CFG.MODE != "prepare":
        raise RuntimeError("Network preparation requires MODE='prepare'")
    started = time.monotonic()
    raw = CFG.data_folder / "raw"
    source = CFG.data_folder / "source_book"
    source.mkdir(parents=True, exist_ok=True)
    files = []
    with httpx.Client(timeout=120, follow_redirects=True) as client:
        for name in CFG.dataset_files:
            url = f"https://huggingface.co/datasets/{CFG.dataset}/resolve/{CFG.dataset_revision}/{name}"
            response = client.get(url)
            response.raise_for_status()
            target = raw / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(response.content)
            files.append({"path": name, "url": url, "sha256": digest_bytes(response.content),
                          "bytes": len(response.content)})
        response = client.get(CFG.source_pdf_url)
        response.raise_for_status()
        if digest_bytes(response.content) != CFG.source_pdf_sha256:
            raise ValueError("The source PDF differs from the reviewed snapshot; revise provenance before continuing")
        (source / "principles-of-financial-accounting.pdf").write_bytes(response.content)
        json_write(source / "manifest.json", {"url": CFG.source_pdf_url, "resolved_url": str(response.url),
                   "sha256": CFG.source_pdf_sha256, "bytes": len(response.content),
                   "attribution": "Christine Jonick; University System of Georgia; University of North Georgia Press, 2017",
                   "license": "CC BY-SA 4.0; see copyright page for third-party exceptions"})
    if not shutil.which("pdftotext"):
        raise RuntimeError("Preparation needs Poppler's pdftotext on PATH; report mode uses the supplied cached extraction")
    subprocess.run(["pdftotext", "-layout", str(source / "principles-of-financial-accounting.pdf"),
                    str(source / "book.txt")], check=True, capture_output=True)
    extractor = subprocess.run(["pdftotext", "-v"], capture_output=True, text=True)
    json_write(CFG.data_folder / "source_manifest.json", {"dataset": CFG.dataset,
               "revision": CFG.dataset_revision, "files": files,
               "preparation_seconds": time.monotonic() - started})
    json_write(source / "extraction_manifest.json", {"pdf_sha256": CFG.source_pdf_sha256,
               "text_sha256": digest_bytes((source / "book.txt").read_bytes()),
               "extractor": (extractor.stdout + extractor.stderr).splitlines()[0],
               "command": "pdftotext -layout principles-of-financial-accounting.pdf book.txt"})


def validate_inputs():
    manifest_path = CFG.data_folder / "source_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError("Cached STAIR evidence is missing. Supply data/stair or explicitly run MODE='prepare'.")
    manifest = json_read(manifest_path)
    if manifest["revision"] != CFG.dataset_revision or manifest["dataset"] != CFG.dataset:
        raise ValueError("Dataset identity/revision differs from the approved snapshot")
    if set(row["path"] for row in manifest["files"]) != set(CFG.dataset_files):
        raise ValueError("Unexpected source-file membership")
    for row in manifest["files"]:
        path = CFG.data_folder / "raw" / row["path"]
        expected_url = f"https://huggingface.co/datasets/{CFG.dataset}/resolve/{CFG.dataset_revision}/{row['path']}"
        if row["url"] != expected_url or digest_bytes(path.read_bytes()) != row["sha256"]:
            raise ValueError(f"Source identity/hash mismatch: {row['path']}")
    source = CFG.data_folder / "source_book"
    if digest_bytes((source / "principles-of-financial-accounting.pdf").read_bytes()) != CFG.source_pdf_sha256:
        raise ValueError("Source PDF hash differs from the reviewed snapshot")
    extraction = json_read(source / "extraction_manifest.json")
    if extraction["pdf_sha256"] != CFG.source_pdf_sha256 or digest_bytes((source / "book.txt").read_bytes()) != extraction["text_sha256"]:
        raise ValueError("Cached source-text extraction failed provenance checks")
    return manifest


def prepare_model_cache():
    if CFG.MODE != "prepare":
        raise RuntimeError("Model downloads require explicit prepare mode")
    started = time.monotonic()
    snapshot_download(CFG.model, revision=CFG.model_revision, local_dir=CFG.model_folder,
                      allow_patterns=["*.json", "spiece.model", "model.safetensors", "README.md"],
                      token=False, max_workers=2)
    files = [{"path": path.name, "sha256": digest_bytes(path.read_bytes()), "bytes": path.stat().st_size}
             for path in sorted(CFG.model_folder.iterdir()) if path.is_file()]
    json_write(CFG.data_folder / "model_manifest.json", {"model": CFG.model, "revision": CFG.model_revision,
               "files": files, "preparation_seconds": time.monotonic() - started, "license": "Apache-2.0"})


def validate_model_cache():
    manifest = json_read(CFG.data_folder / "model_manifest.json")
    if manifest["model"] != CFG.model or manifest["revision"] != CFG.model_revision:
        raise ValueError("Model identity/revision differs from the pinned checkpoint")
    for row in manifest["files"]:
        if digest_bytes((CFG.model_folder / row["path"]).read_bytes()) != row["sha256"]:
            raise ValueError("Model cache checksum failed: " + row["path"])
    return manifest
