from pathlib import Path
from typing import List

def find_existing_training_scripts(repo_root: Path) -> List[Path]:
    """
    Return a list of training scripts in the standard folders (exists only).
    """
    candidates = [
        repo_root / "training" / "lightgbm_return" / "train.py",
        repo_root / "training" / "random_forest_return" / "train.py",
        repo_root / "training" / "lightgbm_up_down" / "train.py",
        repo_root / "training" / "lightgbm_volatility" / "train.py",
        repo_root / "training" / "prophet_forecast" / "train.py",
        repo_root / "training" / "hmm_regime" / "train.py",
    ]
    return [p for p in candidates if p.exists()]
