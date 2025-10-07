from __future__ import annotations

from fastapi import FastAPI

from app.prediction import churn_model

from .schemas import ChurnRequest, ChurnResponse

app = FastAPI(title="Churn Prediction API", version="1.0.0")


@app.get("/")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=ChurnResponse)
def predict_churn(payload: ChurnRequest) -> ChurnResponse:
    result = churn_model.predict(payload.dict())
    return ChurnResponse(**result)
