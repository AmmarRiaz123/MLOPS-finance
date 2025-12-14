"""
Main training script that orchestrates model training.
Can train different model types by importing their respective modules.
"""
import sys
from pathlib import Path

# Add lightgbm_up_down to path for imports
sys.path.append(str(Path(__file__).parent / "lightgbm_up_down"))

def train_lightgbm_up_down():
    """Train LightGBM model for up/down prediction."""
    from lightgbm_up_down.train import main as train_main
    from lightgbm_up_down.evaluate import evaluate_model
    
    print("Training LightGBM Up/Down Prediction Model...")
    
    # Train model
    model, X_test, y_test = train_main()
    
    # Evaluate model
    model_dir = Path(__file__).parent / "lightgbm_up_down" / "model"
    metrics, y_pred, y_proba = evaluate_model(model_dir, X_test, y_test)
    
    return model, metrics

if __name__ == "__main__":
    # Train LightGBM up/down model
    model, metrics = train_lightgbm_up_down()
    
    print("\nTraining pipeline completed!")
    print(f"Final test accuracy: {metrics['accuracy']:.4f}")
