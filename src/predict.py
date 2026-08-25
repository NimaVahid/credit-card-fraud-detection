"""Reusable CLI inference component for Scikit-learn and PyTorch models.

The script follows the project contract:
- load the trained artifacts,
- receive one JSON transaction,
- validate all required features,
- perform prediction with the same preprocessing used during training,
- return JSON output.

No fitting or threshold tuning is performed during inference.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn


class FraudMLP(nn.Module):
    """Simple PyTorch MLP architecture used by the training pipeline."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim_1: int = 64,
        hidden_dim_2: int = 32,
        dropout: float = 0.20,
    ) -> None:
        """Initialize the feed-forward fraud classifier."""

        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim_1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim_1, hidden_dim_2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim_2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw binary-classification logits."""

        return self.network(x).squeeze(1)


def load_artifacts(
    model_path: Path,
    scaler_path: Path,
    encoder_path: Path | None = None,
) -> tuple[dict[str, Any], Any, Any]:
    """Load and validate the saved model, scaler, and optional encoder."""

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model artifact not found: {model_path}. Run src/train.py first."
        )

    if not scaler_path.exists():
        raise FileNotFoundError(
            f"Scaler artifact not found: {scaler_path}. Run src/train.py first."
        )

    artifact = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    encoder = None
    if encoder_path is not None and encoder_path.exists():
        encoder = joblib.load(encoder_path)

    if (
        not isinstance(artifact, dict)
        or artifact.get("artifact_type") != "fraud_detection_model"
    ):
        raise RuntimeError(
            "models/model.pkl is a placeholder or invalid model artifact. "
            "Replace data/creditcard.csv with the real dataset and run "
            "python src/train.py."
        )

    required_keys = {
        "model_kind",
        "model_name",
        "feature_columns",
        "threshold",
        "positive_class",
    }
    missing_keys = sorted(required_keys.difference(artifact))

    if missing_keys:
        raise RuntimeError(
            f"Model artifact is missing required keys: {missing_keys}"
        )

    return artifact, scaler, encoder


def read_json(path: Path) -> dict[str, Any]:
    """Read one transaction from a JSON object."""

    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    if not isinstance(payload, dict):
        raise ValueError("Input JSON must contain one JSON object.")

    return payload


def validate_and_prepare(
    payload: dict[str, Any],
    feature_columns: list[str],
) -> pd.DataFrame:
    """Validate feature schema, numeric values, finiteness, and order."""

    missing = [name for name in feature_columns if name not in payload]
    extra = [name for name in payload if name not in feature_columns]

    if missing:
        raise ValueError(f"Missing features: {missing}")

    if extra:
        raise ValueError(f"Unexpected features: {extra}")

    ordered_values = [payload[name] for name in feature_columns]

    try:
        values = np.asarray(ordered_values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("All feature values must be numeric.") from exc

    if not np.isfinite(values).all():
        raise ValueError("Input contains NaN or infinite values.")

    row = {
        name: float(value)
        for name, value in zip(feature_columns, values)
    }

    return pd.DataFrame([row], columns=feature_columns)


def restore_torch_model(artifact: dict[str, Any]) -> FraudMLP:
    """Rebuild the PyTorch MLP and restore its saved state_dict."""

    config = artifact.get("model_config")
    state_dict = artifact.get("model_state_dict")

    if config is None or state_dict is None:
        raise RuntimeError(
            "PyTorch artifact is missing model_config or model_state_dict."
        )

    model = FraudMLP(**config)
    model.load_state_dict(state_dict)
    model.eval()

    return model


def sklearn_fraud_probability(
    model: Any,
    X: Any,
    positive_class: int,
) -> float:
    """Return class-1 probability from a fitted Scikit-learn estimator."""

    probabilities = model.predict_proba(X)
    classes = np.asarray(model.classes_)
    positions = np.where(classes == positive_class)[0]

    if positions.size != 1:
        raise RuntimeError(
            f"Positive class {positive_class} not found in "
            f"model classes {classes.tolist()}."
        )

    return float(probabilities[0, int(positions[0])])


@torch.no_grad()
def torch_fraud_probability(
    model: FraudMLP,
    X: Any,
) -> float:
    """Return sigmoid fraud probability from the restored PyTorch MLP."""

    X_tensor = torch.tensor(
        np.asarray(X, dtype=np.float32),
        dtype=torch.float32,
    )

    model.eval()
    logits = model(X_tensor)
    probability = torch.sigmoid(logits)

    return float(probability.item())


def predict_transaction(
    payload: dict[str, Any],
    artifact: dict[str, Any],
    scaler: Any,
    encoder: Any = None,
) -> dict[str, Any]:
    """Run one validated transaction through the frozen inference pipeline."""

    if encoder is not None:
        raise RuntimeError(
            "An encoder artifact was supplied, but this numeric dataset does "
            "not require categorical encoding."
        )

    feature_columns = list(artifact["feature_columns"])
    threshold = float(artifact["threshold"])
    positive_class = int(artifact["positive_class"])

    X_new = validate_and_prepare(payload, feature_columns)

    if scaler is not None:
        X_for_model = scaler.transform(X_new)
    else:
        X_for_model = X_new

    if artifact["model_kind"] == "sklearn":
        probability = sklearn_fraud_probability(
            artifact["model"],
            X_for_model,
            positive_class,
        )

    elif artifact["model_kind"] == "pytorch":
        model = restore_torch_model(artifact)
        probability = torch_fraud_probability(
            model,
            X_for_model,
        )

    else:
        raise RuntimeError(
            f"Unsupported model_kind: {artifact['model_kind']}"
        )

    class_id = int(probability >= threshold)
    label = "Fraud" if class_id == 1 else "Legitimate"

    return {
        "prediction": label,
        "class_id": class_id,
        "probability": probability,
        "threshold": threshold,
        "status": "success",
    }


def parse_args() -> argparse.Namespace:
    """Parse command-line inference arguments."""

    root = Path(__file__).resolve().parent.parent

    parser = argparse.ArgumentParser(
        description="Predict fraud for one JSON transaction."
    )

    parser.add_argument(
        "input_json",
        type=Path,
        help="Path to a JSON file containing exactly one transaction.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path for JSON output.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=root / "models" / "model.pkl",
    )
    parser.add_argument(
        "--scaler",
        type=Path,
        default=root / "models" / "scaler.pkl",
    )
    parser.add_argument(
        "--encoder",
        type=Path,
        default=root / "models" / "encoder.pkl",
    )

    return parser.parse_args()


def main() -> None:
    """Run CLI prediction and print JSON output."""

    args = parse_args()

    artifact, scaler, encoder = load_artifacts(
        args.model,
        args.scaler,
        args.encoder,
    )

    payload = read_json(args.input_json)

    result = predict_transaction(
        payload,
        artifact,
        scaler,
        encoder,
    )

    output_text = json.dumps(
        result,
        indent=4,
        ensure_ascii=False,
    )

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output_text + "\n", encoding="utf-8")

    print(output_text)


if __name__ == "__main__":
    main()
