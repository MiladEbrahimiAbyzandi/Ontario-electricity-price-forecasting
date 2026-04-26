from pathlib import Path
from dataclasses import dataclass, field

# --- Paths -------------------------------------------------------------------
BASE_DIR   = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / 'artifacts' / 'model'
SCALER_DIR = BASE_DIR / 'artifacts' / 'scalers'


# --- Config dataclasses ------------------------------------------------------

@dataclass
class DataConfig:
    feature_cols: list = field(default_factory=lambda: [
        "hoep", "market_demand", "ontario_demand",
        "nuclear", "gas", "hydro", "wind", "solar", "biofuel",
    ])
    target_col: str  = "hoep"
    seq_length: int  = 168
    horizons: list   = field(default_factory=lambda: [1, 2, 3])

    @property
    def input_size(self) -> int:
        return len(self.feature_cols)


@dataclass
class TrainConfig:
    hidden_size: int    = 32
    num_layers: int     = 1
    dropout: float      = 0.2
    learning_rate: float = 1e-2
    batch_size: int     = 32
    epochs: int         = 100
    patience: int       = 10


DATA_CFG  = DataConfig()
TRAIN_CFG = TrainConfig()
