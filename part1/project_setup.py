"""Shared configuration and reproducibility records for the Project 1 notebook.

Run from the repository root with:
    .venv/bin/python -m part1.project_setup

This prepares local folders and records the environment. It does not download
SNLI, fit preprocessing, or train classifiers.
"""

import argparse
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
from typing import Any

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "part1/configs/snli_project1.json"
ARTIFACT_FOLDERS = (
    "manifests", "figures", "metrics", "predictions", "checkpoints", "folds", "features"
)


def set_random_seeds(seed: int) -> None:
    """Seed the global generators used by the tutorial's training code.

    Dedicated generators, CV splits, and sampling still need their own explicit
    seeds. This does not promise identical results across platforms/devices.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _git_output(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments], cwd=REPO_ROOT, check=True,
        capture_output=True, text=True,
    ).stdout.strip()


def prepare_workspace(config_path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    """Read the config, seed libraries, and capture a local setup manifest.

    Configuration paths are relative to the repository root. Returned artifact
    paths are absolute so the notebook also works when launched in ``part1``.
    Rerunning replaces the latest setup manifest; later training runs will need
    their own manifests rather than using this setup record as a run history.
    """
    config_path = Path(config_path)
    if not config_path.is_absolute():
        config_path = REPO_ROOT / config_path
    config_bytes = config_path.read_bytes()
    config = json.loads(config_bytes)
    set_random_seeds(config["seeds"]["model"])

    output_dir = REPO_ROOT / config["paths"]["output_dir"]
    data_dir = REPO_ROOT / config["paths"]["data_dir"]
    paths = {"output": output_dir, "data": data_dir}
    paths.update({name: output_dir / name for name in ARTIFACT_FOLDERS})
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)

    environment = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"], check=True,
        capture_output=True, text=True,
    ).stdout
    environment_path = paths["manifests"] / "environment.txt"
    environment_path.write_text(environment, encoding="utf-8")

    git_status = _git_output("status", "--porcelain")
    manifest = {
        "stage": "workspace_setup",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_name": config["experiment_name"],
        "configuration": config,
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "dataset_loaded": False,
        "python": {
            "executable": sys.executable,
            "version": platform.python_version(),
            "prefix": sys.prefix,
        },
        "hardware": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "logical_cpu_count": os.cpu_count(),
            "configured_device": config["device"],
            "cuda_available": torch.cuda.is_available(),
        },
        "packages": {
            distribution.metadata["Name"]: distribution.version
            for distribution in metadata.distributions()
        },
        "repository": {
            "commit": _git_output("rev-parse", "HEAD"),
            "dirty": bool(git_status),
            "status": git_status.splitlines(),
        },
        "artifacts": {
            "setup_manifest": str(paths["manifests"] / "setup.json"),
            "environment": str(environment_path),
        },
    }
    manifest_path = Path(manifest["artifacts"]["setup_manifest"])
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {"config": config, "paths": paths, "manifest": manifest}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    arguments = parser.parse_args()
    workspace = prepare_workspace(arguments.config)
    print("Python:", workspace["manifest"]["python"]["executable"])
    print("Device:", workspace["config"]["device"])
    print("Dataset revision:", workspace["config"]["dataset"]["revision"])
    print("Manifest:", workspace["manifest"]["artifacts"]["setup_manifest"])
