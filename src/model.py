import torch
import torch.nn as nn
from src.config import INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, DROPOUT

class GRURegressor(nn.Module):
    """
    GRU-based regressor for HOEP forecasting.
    It is identical to the model trained in notebook 06
    Architecture: Input -> GRU -> Linear(1)
    """
    def __init__(self,
                 input_size: int = INPUT_SIZE,
                 hidden_size: int = HIDDEN_SIZE,
                 num_layers: int = NUM_LAYERS,
                 dropout: float = DROPOUT
                 ):
        super().__init__()

        effective_dropout = dropout if num_layers > 1 else 0.0

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=effective_dropout
            )
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size,1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        out = out[:, -1, :]
        out = self.dropout(out)
        out = self.fc(out)
        return out