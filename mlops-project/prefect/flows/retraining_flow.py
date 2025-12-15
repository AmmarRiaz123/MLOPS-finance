from pathlib import Path
import subprocess
import logging
from typing import List, Dict, Any, Optional
import importlib.util
import importlib
import sys

from prefect import flow, task

# robust import of local download task
try:
    # preferred: package-relative import when running as package
    from ..tasks.download_data import download_symbol
except Exception:
    # fallback: dynamically load the task module from file path so script can be run directly
    tasks_path = Path(__file__).resolve().parents[1] / "tasks" / "download_data.py"
    spec = importlib.util.spec_from_file_location("local_prefect_tasks_download_data", str(tasks_path))
    module = importlib.util.module_from_spec(spec)
    sys.modules["local_prefect_tasks_download_data"] = module
    spec.loader.exec_module(module)
    download_symbol = getattr(module, "download_symbol")

LOG = logging.getLogger("retraining_flow")
LOG.setLevel(logging.INFO)


@task
def _run_training_script(script_path: Path, python_exe: Optional[str] = None) -> Dict[str, Any]:
    """
    Run a single training script via subprocess. Return dict with status and error (if any).
    This task is lightweight so Prefect can track it and we get clear logs.
    """
    python_cmd = python_exe or "python"
    LOG.info("Running training script: %s", script_path)
    try:
        res = subprocess.run([python_cmd, str(script_path)], check=True, capture_output=True, text=True)
        LOG.info("Completed: %s", script_path)
        return {"script": str(script_path), "ok": True, "stdout": res.stdout, "stderr": res.stderr}
    except subprocess.CalledProcessError as exc:
        LOG.error("Training failed for %s: %s", script_path, exc.stderr or exc)
        return {"script": str(script_path), "ok": False, "stdout": getattr(exc, "stdout", ""), "stderr": getattr(exc, "stderr", str(exc))}


@flow(name="retraining-flow")
def retraining_flow(symbols: Optional[List[str]] = None,
                    force_download: bool = False,
                    force_retrain: bool = False,
                    python_exe: Optional[str] = None) -> Dict[str, Any]:
    """
    Prefect flow to download latest data and run training scripts sequentially.

    Args:
      symbols: list of ticker symbols to download (default: ["IBM"])
      force_download: if True, re-download even if files exist (passed to task as needed)
      force_retrain: currently reserved for future behavior (flow will always attempt training)
      python_exe: optional python executable to run training scripts (e.g. sys.executable)
    Returns:
      summary dict with per-script status.
    """
    repo_root = Path(__file__).resolve().parents[2]  # mlops-project
    LOG.info("Starting retraining flow (repo_root=%s)", repo_root)

    symbols = symbols or ["IBM"]

    # 1) Download symbols (sequential)
    downloaded_files = []
    for s in symbols:
        LOG.info("Downloading symbol: %s", s)
        try:
            # call the existing Prefect task - this will execute the download logic
            out_path = download_symbol(s)  # calling task directly executes it
            LOG.info("Downloaded %s -> %s", s, out_path)
            downloaded_files.append(out_path)
        except Exception as e:
            LOG.error("Download failed for %s: %s", s, e)
            downloaded_files.append({"symbol": s, "error": str(e)})

    # 2) Define training script paths (relative to repo root)
    training_scripts = [
        repo_root / "training" / "lightgbm_return" / "train.py",
        repo_root / "training" / "random_forest_return" / "train.py",
        repo_root / "training" / "lightgbm_up_down" / "train.py",
        repo_root / "training" / "lightgbm_volatility" / "train.py",
        repo_root / "training" / "prophet_forecast" / "train.py",
        repo_root / "training" / "market_regime_hmm" / "train.py",
    ]

    # 3) Run each training script sequentially; collect results even if some fail
    results = []
    for script in training_scripts:
        if not script.exists():
            LOG.warning("Training script not found, skipping: %s", script)
            results.append({"script": str(script), "ok": False, "error": "not found"})
            continue
        r = _run_training_script(script, python_exe=python_exe)
        results.append(r)

    # Compose summary
    summary = {
        "downloaded": downloaded_files,
        "training_results": results,
        "success_count": sum(1 for r in results if r.get("ok")),
        "failure_count": sum(1 for r in results if not r.get("ok")),
    }

    LOG.info("Retraining flow completed: %s", summary)
    return summary


# optional local runner
if __name__ == "__main__":
    import sys as _sys
    logging.basicConfig(level=logging.INFO)
    out = retraining_flow(symbols=["IBM"], force_download=False, python_exe=_sys.executable)
    print(out)
