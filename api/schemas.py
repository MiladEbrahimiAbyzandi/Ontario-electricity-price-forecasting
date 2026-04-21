import numpy as np
from pydantic import BaseModel, Field, model_validator


class ForecastRequest(BaseModel):
    """
    The caller must send 168 rows x 9 columns of recent hourly ontario electricity price.
    Each row is one hour, ordered oldest to newest.
    """

    recent_data: list[list[float]] = Field(
        ...,
        description=(
            "168 hours x 9 features in order:"
            "hoep, market_demand, ontario_demand, "
            "nuclear, gas, hydro, wind, solar, biofuel"
        ))
    horizon : int = Field(..., ge=1, le=3,
                          description=(
                              "Forcast horizon: 1, 2, or 3 hours ahead"
                          ))
    
    @model_validator(mode="after")
    def check_shape(self):
        rows = len(self.recent_data)
        cols = len(self.recent_data[0])
        if rows != 168 or cols != 9:
            raise ValueError(
                f"expected (168, 9), got ({rows}, {cols})"
            )
        return self
    
    def to_numpy(self) -> np.ndarray:
        return np.array(self.recent_data, dtype=np.float32)
    
class ForecastResponse (BaseModel):
    horizon: int 
    predicted_hoep: float = Field(
        ...,
        description=(" Predicted price in $/MWh")
    )
