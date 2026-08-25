"""Optional FastAPI prediction service for the fraud mini-project.

The REST API reuses `predict_transaction()` from predict.py, so CLI and API
inference share exactly the same preprocessing, feature validation, model,
and threshold logic.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, ConfigDict

from src.predict import load_artifacts, predict_transaction


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "model.pkl"
SCALER_PATH = PROJECT_ROOT / "models" / "scaler.pkl"
ENCODER_PATH = PROJECT_ROOT / "models" / "encoder.pkl"


class TransactionRequest(BaseModel):
    """Strict request schema containing all 30 model features."""

    model_config = ConfigDict(extra="forbid")

    Time: float
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float


class PredictionResponse(BaseModel):
    """Response schema matching the project prediction contract."""

    prediction: str
    class_id: int
    probability: float
    threshold: float
    status: str


class HealthResponse(BaseModel):
    """Readiness response for the optional REST service."""

    status: str
    model_loaded: bool
    detail: str | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model artifacts once when the API process starts."""

    app.state.artifact = None
    app.state.scaler = None
    app.state.encoder = None
    app.state.load_error = None

    try:
        artifact, scaler, encoder = load_artifacts(
            MODEL_PATH,
            SCALER_PATH,
            ENCODER_PATH,
        )
        app.state.artifact = artifact
        app.state.scaler = scaler
        app.state.encoder = encoder

    except Exception as exc:
        # Keep /health available even before the real model has been trained.
        app.state.load_error = str(exc)

    yield

    app.state.artifact = None
    app.state.scaler = None
    app.state.encoder = None


app = FastAPI(
    title="Credit Card Fraud Detection API",
    version="1.0.0",
    description=(
        "Optional FastAPI extension for the credit-card fraud mini-project."
    ),
    lifespan=lifespan,
)


@app.get(
    "/health",
    response_model=HealthResponse,
)
def health() -> HealthResponse:
    """Return whether trained model artifacts are ready."""

    if app.state.artifact is None:
        return HealthResponse(
            status="not_ready",
            model_loaded=False,
            detail=app.state.load_error,
        )

    return HealthResponse(
        status="healthy",
        model_loaded=True,
    )


@app.post(
    "/predict",
    response_model=PredictionResponse,
)
def predict(
    transaction: TransactionRequest,
) -> PredictionResponse:
    """Validate and score one transaction with the frozen pipeline."""

    if app.state.artifact is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                app.state.load_error
                or "Model artifacts are not ready."
            ),
        )

    try:
        result = predict_transaction(
            transaction.model_dump(),
            app.state.artifact,
            app.state.scaler,
            app.state.encoder,
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        # In production, log the internal exception server-side.
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Prediction failed.",
        ) from exc

    return PredictionResponse(**result)
