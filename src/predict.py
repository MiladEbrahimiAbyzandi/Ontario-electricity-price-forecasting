
import numpy as np
import torch
import joblib
from src.config import (
    DATA_CFG,
    TRAIN_CFG,
    MODELS_DIR,
    SCALER_DIR,
)
from src.model import GRURegressor

# ── Device ─────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Scalers — loaded once at startup ───────────────────────────────────
scaler_X = joblib.load(SCALER_DIR / "scaler_X.pkl")
scaler_y = joblib.load(SCALER_DIR / "scaler_y.pkl")


def _load_model(horizon: int) -> GRURegressor:
    """
    Load the champion model for the given horizon from local artifacts.
    Models are versioned with DVC — run 'dvc pull' to get latest champion.
    """
    model = GRURegressor(
        input_size=DATA_CFG.input_size,
        hidden_size=TRAIN_CFG.hidden_size,
        num_layers=TRAIN_CFG.num_layers,
        dropout=TRAIN_CFG.dropout,
    )
    model_path = MODELS_DIR / f"gru_h{horizon}.pt"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}\n"
            f"Run 'dvc pull' to download model weights, "
            f"or run 'python -m src.train' to train from scratch."
        )

    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model.to(device)

models = {h: _load_model(h) for h in DATA_CFG.horizons}

def predict(recent_data:np.ndarray, horizon:int) -> float:

    """
    Make a forecast for a given horizon.

    Parameters
    ----------
    recent_data : np.ndarray, shape (168, 9)
        The last 168 hours of the 9 feature columns,
        in the same order as FEATURE_COLS — NOT yet scaled.
    horizon : int
        1, 2, or 3 hours ahead.

    Returns
    -------
    float
        Predicted HOEP in the original price units ($/MWh).
    """

    if recent_data.shape != (DATA_CFG.seq_length, DATA_CFG.input_size):
        raise ValueError(
            f"Expected shape ({DATA_CFG.seq_length}, {DATA_CFG.input_size}), "
            f"got {recent_data.shape}"
        )
    if horizon not in DATA_CFG.horizons:
        raise ValueError(
            f"horizon must be one of {DATA_CFG.horizons}"
        )
    X_scaled = scaler_X.transform(recent_data)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        y_scaled = models[horizon](X_tensor).cpu().numpy()

    y_pred = scaler_y.inverse_transform(y_scaled).flatten()[0]

    return float(y_pred)

