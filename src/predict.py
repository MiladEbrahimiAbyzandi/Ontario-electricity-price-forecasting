
import numpy as np
import joblib
from src.model import GRURegressor
import torch
from src.config import (HIDDEN_SIZE,INPUT_SIZE,NUM_LAYERS,
                        MODELS_DIR,SEQ_LENGTH,HORIZONS,
                        SCALER_DIR)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
X_scaler = joblib.load(SCALER_DIR/"scaler_X.pkl")
y_scaler = joblib.load(SCALER_DIR/"scaler_y.pkl")


def _load_model (horizon: int) -> GRURegressor:
    """Load the trained GRU model for the given forecast horizon"""
    
    model = GRURegressor(
        input_size = INPUT_SIZE,
        hidden_size= HIDDEN_SIZE,
        num_layers= NUM_LAYERS,
        dropout= False
    )

    state_dict = torch.load(
        MODELS_DIR/f"gru_h{horizon}.pt",
        map_location=device
    )

    model.load_state_dict(state_dict)
    model.eval()

    return model.to(device)

models={h: _load_model(h) for h in HORIZONS}

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

    if recent_data.shape != (SEQ_LENGTH, INPUT_SIZE):
        raise ValueError(
            f"Expected shape ({SEQ_LENGTH}, {INPUT_SIZE}), "
            f"got {recent_data.shape}"
        )
    if horizon not in HORIZONS:
        raise ValueError(
            f"horizon must be one of {HORIZONS}"
        )
    X_scaled = X_scaler.transform(recent_data)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        y_scaled = models[horizon](X_tensor).cpu().numpy()

    y_pred = y_scaler.inverse_transform(y_scaled).flatten()[0]

    return float(y_pred)

