"""
Main training script that orchestrates model training.
Can train different model types by importing their respective modules.
"""
import sys
from pathlib import Path
import argparse
import importlib.util
import json

# Add lightgbm_up_down to path for imports (existing behavior)
sys.path.append(str(Path(__file__).parent / "lightgbm_up_down"))

def train_lightgbm_up_down():
    """Train LightGBM model for up/down prediction (existing behavior)."""
    from lightgbm_up_down.train import main as train_main
    from lightgbm_up_down.evaluate import evaluate_model
    
    print("Training LightGBM Up/Down Prediction Model...")
    model, X_test, y_test = train_main()
    print("Evaluating LightGBM model...")
    model_dir = Path(__file__).parent / "lightgbm_up_down" / "model"
    metrics, y_pred, y_proba = evaluate_model(model_dir, X_test, y_test)
    return {"model": "lightgbm_up_down", "metrics": metrics}

def _load_module_from_path(name: str, filepath: Path):
    """Dynamically import a module from a file path."""
    spec = importlib.util.spec_from_file_location(name, str(filepath))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def train_prophet_forecast(repo_root: Path):
    """Train and evaluate the Prophet forecasting pipeline using file-based imports."""
    mod_dir = repo_root / "training" / "prophet_forecast"
    train_path = mod_dir / "train.py"
    eval_path = mod_dir / "evaluate.py"

    # import train module
    train_mod = _load_module_from_path("prophet_forecast.train", train_path)
    print("Training Prophet forecasting model...")
    # train() returns metadata dict
    train_meta = train_mod.train()
    # import evaluate module and run evaluation
    eval_mod = _load_module_from_path("prophet_forecast.evaluate", eval_path)
    print("Evaluating Prophet forecasting model...")
    metrics = eval_mod.evaluate()
    return {"model": "prophet_forecast", "train_meta": train_meta, "metrics": metrics}

def main():
    parser = argparse.ArgumentParser(description="Train pipelines")
    parser.add_argument("--model", choices=["lightgbm", "prophet", "all"], default="all",
                        help="Which model pipeline to run")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent

    results = []
    if args.model in ("lightgbm", "all"):
        try:
            res = train_lightgbm_up_down()
            results.append(res)
        except Exception as e:
            print(f"LightGBM pipeline failed: {e}")

    if args.model in ("prophet", "all"):
        try:
            res = train_prophet_forecast(repo_root)
            results.append(res)
        except Exception as e:
            print(f"Prophet pipeline failed: {e}")

    # simple summary
    summary = {r["model"]: (r.get("metrics") or r.get("train_meta")) for r in results}
    print("\n--- Pipeline summary ---")
    print(json.dumps(summary, indent=2, default=str))

if __name__ == "__main__":
    main()
