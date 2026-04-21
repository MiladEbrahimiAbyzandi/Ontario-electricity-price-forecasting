from fastapi import FastAPI, HTTPException
from api.schemas import ForecastRequest, ForecastResponse
from src.predict import predict
app = FastAPI(
    title= "HOEP forecasting API",
    description= "GRU-based forecasting of Ontario hourly electricity prices",
    version= "0.1.0"
)

@app.get("/health")
def health():
    """Simple check that API is running well."""
    return {"status":"ok"}

@app.post("/predict", response_model = ForecastResponse)
def forecast(request: ForecastRequest):
    """Accepts 168 hours of recent data along with forecasting horizon and returns forecasted price"""
    
    data = request.to_numpy()
    
    try:
        price = predict(recent_data=data, horizon=request.horizon)

    except Exception as e:
        raise HTTPException(status_code=500, detail= str(e))

    return ForecastResponse(
        horizon=request.horizon,
        predicted_hoep=round(price, 2)
    )

