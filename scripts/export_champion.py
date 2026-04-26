"""
After reviewing runs in MLflow UI:
1. Register the best version for each horizon manually
2. Set @champion alias on the best version
3. Run this script to export champion weights to models/

Usage:
    python scripts/export_champion.py --tracking-uri http://localhost:5000
"""
import torch
import mlflow
import mlflow.pytorch
from src.config import MODELS_DIR, DATA_CFG


def export_champion(horizon: int) -> None:
    """
    Load @champion model for given horizon from MLflow
    and save weights to models/gru_h{horizon}.pt
    """
    model_name = f"hoep-gru-h{horizon}"
    model_uri  = f"models:/{model_name}@champion"

    print(f"\nExporting horizon {horizon}...")

    try:
        model = mlflow.pytorch.load_model(model_uri,
                                          map_location=torch.device("cpu"))
    except Exception:
        raise RuntimeError(
            f"Could not load @champion for {model_name}.\n"
            f"Go to MLflow UI → Models → {model_name} → "
            f"pick best version → Add alias → 'champion'"
        )

    dst_path = MODELS_DIR / f"gru_h{horizon}.pt"
    torch.save(model.state_dict(), dst_path)
    print(f"Saved → {dst_path} ✅")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Export @champion models from MLflow to models/"
    )
    parser.add_argument(
        "--tracking-uri",
        default="http://localhost:5000",
        help="MLflow tracking URI",
    )
    args = parser.parse_args()

    mlflow.set_tracking_uri(args.tracking_uri)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    for h in DATA_CFG.horizons:
        export_champion(h)

    print("\nAll champion models exported successfully.")