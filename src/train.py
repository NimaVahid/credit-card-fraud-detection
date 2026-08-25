"""Complete training workflow for the fraud-detection assignment.

Implemented requirements
------------------------
- dataset structure analysis
- descriptive statistics
- missing/infinite/duplicate checks
- stratified train/test split
- leakage-safe scaling
- Logistic Regression
- KNN
- Decision Tree
- bonus PyTorch MLP
- Accuracy, Precision, Recall, F1, and Confusion Matrix for every model
- 5-Fold Stratified Cross Validation
- mandatory KNN scaling experiment
- mandatory Decision Tree max_depth experiment
- mandatory threshold experiment at 0.3, 0.5, and 0.7
- final model and threshold selection
- untouched final test evaluation
- model/scaler/encoder persistence
- prediction smoke test
- automatic experiments.md generation
- automatic README "After Training Analysis" update

The final test set is never used to make a development decision.
"""

from __future__ import annotations

import argparse
import copy
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from data_prep import (
    FEATURE_COLUMNS,
    dataset_structure,
    dataset_summary,
    descriptive_statistics,
    load_dataset,
    split_features_target,
    stratified_train_test_split,
)


RANDOM_STATE = 42
TEST_SIZE = 0.20
N_SPLITS = 5
DEFAULT_THRESHOLD = 0.50
THRESHOLDS = (0.30, 0.50, 0.70)
TREE_DEPTHS = (2, 5, 10, None)

MLP_CONFIG = {
    "input_dim": len(FEATURE_COLUMNS),
    "hidden_dim_1": 64,
    "hidden_dim_2": 32,
    "dropout": 0.20,
}
MLP_EPOCHS = 12
MLP_BATCH_SIZE = 2048
MLP_LEARNING_RATE = 1e-3


@dataclass
class FoldResult:
    """Store one fold's validation metrics and optional training F1."""

    accuracy: float
    precision: float
    recall: float
    f1: float
    train_f1: float | None = None


@dataclass
class CVResult:
    """Store CV fold metrics plus training out-of-fold probabilities."""

    fold_results: list[FoldResult]
    oof_probability: np.ndarray

    @property
    def mean_accuracy(self) -> float:
        """Return mean validation Accuracy."""

        return float(np.mean([row.accuracy for row in self.fold_results]))

    @property
    def mean_precision(self) -> float:
        """Return mean validation Precision."""

        return float(np.mean([row.precision for row in self.fold_results]))

    @property
    def mean_recall(self) -> float:
        """Return mean validation Recall."""

        return float(np.mean([row.recall for row in self.fold_results]))

    @property
    def mean_f1(self) -> float:
        """Return mean validation F1."""

        return float(np.mean([row.f1 for row in self.fold_results]))

    @property
    def f1_std(self) -> float:
        """Return F1 standard deviation across folds."""

        return float(np.std([row.f1 for row in self.fold_results]))

    @property
    def mean_train_f1(self) -> float | None:
        """Return mean Train F1 when training scores were collected."""

        scores = [
            row.train_f1
            for row in self.fold_results
            if row.train_f1 is not None
        ]
        return float(np.mean(scores)) if scores else None

    @property
    def generalization_gap(self) -> float | None:
        """Return Train F1 minus Validation F1."""

        if self.mean_train_f1 is None:
            return None
        return self.mean_train_f1 - self.mean_f1


class FraudMLP(nn.Module):
    """Simple bonus PyTorch neural network for binary fraud classification."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim_1: int = 64,
        hidden_dim_2: int = 32,
        dropout: float = 0.20,
    ) -> None:
        """Initialize the feed-forward network."""

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
        """Return raw logits for BCEWithLogitsLoss."""

        return self.network(x).squeeze(1)


def seed_everything(seed: int = RANDOM_STATE) -> None:
    """Set Python, NumPy, and PyTorch random seeds."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def metric_dict(
    y_true: pd.Series | np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """Compute Accuracy, Precision, Recall, and F1."""

    return {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Precision": float(
            precision_score(y_true, y_pred, pos_label=1, zero_division=0)
        ),
        "Recall": float(
            recall_score(y_true, y_pred, pos_label=1, zero_division=0)
        ),
        "F1": float(
            f1_score(y_true, y_pred, pos_label=1, zero_division=0)
        ),
    }


def confusion_values(
    y_true: pd.Series | np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, int]:
    """Return TN, FP, FN, and TP with an explicit label order."""

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    ).ravel()

    return {
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }


def sklearn_positive_probability(
    estimator: Any,
    X: Any,
) -> np.ndarray:
    """Return fraud-class probability from a fitted Scikit-learn estimator."""

    probabilities = estimator.predict_proba(X)
    classes = np.asarray(estimator.classes_)
    positions = np.where(classes == 1)[0]

    if positions.size != 1:
        raise RuntimeError(
            f"Fraud class 1 not present in estimator classes: {classes.tolist()}"
        )

    return probabilities[:, int(positions[0])]


def evaluate_sklearn_cv(
    estimator: Any,
    X: pd.DataFrame,
    y: pd.Series,
    cv: StratifiedKFold,
    *,
    include_train_f1: bool = True,
) -> CVResult:
    """Run leakage-safe 5-fold stratified CV for a Scikit-learn estimator."""

    oof_probability = np.empty(len(X), dtype=float)
    folds: list[FoldResult] = []

    for train_idx, validation_idx in cv.split(X, y):
        fold_estimator = clone(estimator)

        X_train_fold = X.iloc[train_idx]
        y_train_fold = y.iloc[train_idx]
        X_validation_fold = X.iloc[validation_idx]
        y_validation_fold = y.iloc[validation_idx]

        fold_estimator.fit(X_train_fold, y_train_fold)

        validation_probability = sklearn_positive_probability(
            fold_estimator,
            X_validation_fold,
        )
        oof_probability[validation_idx] = validation_probability

        validation_pred = (
            validation_probability >= DEFAULT_THRESHOLD
        ).astype(int)

        scores = metric_dict(y_validation_fold, validation_pred)

        train_f1 = None
        if include_train_f1:
            train_probability = sklearn_positive_probability(
                fold_estimator,
                X_train_fold,
            )
            train_pred = (
                train_probability >= DEFAULT_THRESHOLD
            ).astype(int)
            train_f1 = float(
                f1_score(
                    y_train_fold,
                    train_pred,
                    pos_label=1,
                    zero_division=0,
                )
            )

        folds.append(
            FoldResult(
                accuracy=scores["Accuracy"],
                precision=scores["Precision"],
                recall=scores["Recall"],
                f1=scores["F1"],
                train_f1=train_f1,
            )
        )

    return CVResult(
        fold_results=folds,
        oof_probability=oof_probability,
    )


def class_pos_weight(y: pd.Series | np.ndarray) -> float:
    """Return negative/positive count ratio for BCEWithLogitsLoss."""

    values = np.asarray(y, dtype=int)
    positives = int(np.sum(values == 1))
    negatives = int(np.sum(values == 0))

    if positives == 0:
        raise ValueError("PyTorch training fold contains no fraud samples.")

    return negatives / positives


def train_torch_model(
    X_scaled: np.ndarray,
    y: pd.Series | np.ndarray,
    *,
    device: torch.device,
    epochs: int = MLP_EPOCHS,
) -> FraudMLP:
    """Train the PyTorch MLP using weighted binary cross-entropy."""

    seed_everything()

    X_tensor = torch.tensor(
        np.asarray(X_scaled, dtype=np.float32),
        dtype=torch.float32,
    )
    y_tensor = torch.tensor(
        np.asarray(y, dtype=np.float32),
        dtype=torch.float32,
    )

    loader = DataLoader(
        TensorDataset(X_tensor, y_tensor),
        batch_size=MLP_BATCH_SIZE,
        shuffle=True,
        generator=torch.Generator().manual_seed(RANDOM_STATE),
    )

    model = FraudMLP(**MLP_CONFIG).to(device)

    pos_weight = torch.tensor(
        [class_pos_weight(y)],
        dtype=torch.float32,
        device=device,
    )

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=MLP_LEARNING_RATE,
    )

    for _ in range(epochs):
        model.train()

        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

    return model


@torch.no_grad()
def torch_probabilities(
    model: FraudMLP,
    X_scaled: np.ndarray,
    *,
    device: torch.device,
) -> np.ndarray:
    """Return sigmoid probabilities from a fitted PyTorch MLP."""

    model.eval()

    X_tensor = torch.tensor(
        np.asarray(X_scaled, dtype=np.float32),
        dtype=torch.float32,
    )

    loader = DataLoader(
        TensorDataset(X_tensor),
        batch_size=8192,
        shuffle=False,
    )

    parts: list[np.ndarray] = []

    for (X_batch,) in loader:
        logits = model(X_batch.to(device))
        probabilities = torch.sigmoid(logits).cpu().numpy()
        parts.append(probabilities)

    return np.concatenate(parts)


def evaluate_torch_cv(
    X: pd.DataFrame,
    y: pd.Series,
    cv: StratifiedKFold,
) -> CVResult:
    """Run leakage-safe 5-fold stratified CV for the bonus PyTorch MLP."""

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    oof_probability = np.empty(len(X), dtype=float)
    folds: list[FoldResult] = []

    for fold_no, (train_idx, validation_idx) in enumerate(
        cv.split(X, y),
        start=1,
    ):
        print(
            f"  PyTorch MLP fold {fold_no}/{cv.n_splits} "
            f"using {device.type}"
        )

        X_train_fold = X.iloc[train_idx]
        y_train_fold = y.iloc[train_idx]
        X_validation_fold = X.iloc[validation_idx]
        y_validation_fold = y.iloc[validation_idx]

        # Critical leakage rule: fit the scaler only on this fold's training data.
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_fold)
        X_validation_scaled = scaler.transform(X_validation_fold)

        model = train_torch_model(
            X_train_scaled,
            y_train_fold,
            device=device,
        )

        validation_probability = torch_probabilities(
            model,
            X_validation_scaled,
            device=device,
        )
        oof_probability[validation_idx] = validation_probability

        validation_pred = (
            validation_probability >= DEFAULT_THRESHOLD
        ).astype(int)

        validation_scores = metric_dict(
            y_validation_fold,
            validation_pred,
        )

        train_probability = torch_probabilities(
            model,
            X_train_scaled,
            device=device,
        )
        train_pred = (
            train_probability >= DEFAULT_THRESHOLD
        ).astype(int)
        train_f1 = float(
            f1_score(
                y_train_fold,
                train_pred,
                pos_label=1,
                zero_division=0,
            )
        )

        folds.append(
            FoldResult(
                accuracy=validation_scores["Accuracy"],
                precision=validation_scores["Precision"],
                recall=validation_scores["Recall"],
                f1=validation_scores["F1"],
                train_f1=train_f1,
            )
        )

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return CVResult(
        fold_results=folds,
        oof_probability=oof_probability,
    )


def logistic_candidate() -> Pipeline:
    """Return Logistic Regression with fold-local StandardScaler."""

    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    max_iter=1000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def knn_candidate(*, scaled: bool) -> Any:
    """Return KNN with or without scaling for the controlled experiment."""

    knn = KNeighborsClassifier(
        n_neighbors=5,
        weights="uniform",
        n_jobs=1,
    )

    if not scaled:
        return knn

    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", knn),
        ]
    )


def tree_candidate(
    depth: int | None,
) -> DecisionTreeClassifier:
    """Return a Decision Tree with the requested maximum depth."""

    return DecisionTreeClassifier(
        max_depth=depth,
        random_state=RANDOM_STATE,
    )


def cv_mean_table(
    results: dict[str, CVResult],
) -> pd.DataFrame:
    """Create the required mean CV metric comparison table."""

    rows = []

    for name, result in results.items():
        rows.append(
            {
                "Model": name,
                "Mean Precision": result.mean_precision,
                "Mean Recall": result.mean_recall,
                "Mean F1": result.mean_f1,
                "F1 STD": result.f1_std,
                "Train F1": result.mean_train_f1,
                "Generalization Gap": result.generalization_gap,
            }
        )

    return pd.DataFrame(rows)


def oof_model_evaluation(
    results: dict[str, CVResult],
    y_train: pd.Series,
) -> tuple[pd.DataFrame, dict[str, dict[str, int]]]:
    """Report all required metrics and confusion matrices for every model.

    These are aggregated out-of-fold training evaluations, so model selection
    remains independent of the final test set.
    """

    metric_rows = []
    matrices: dict[str, dict[str, int]] = {}

    for name, result in results.items():
        y_pred = (
            result.oof_probability >= DEFAULT_THRESHOLD
        ).astype(int)

        row = {
            "Model": name,
            **metric_dict(y_train, y_pred),
        }
        metric_rows.append(row)
        matrices[name] = confusion_values(y_train, y_pred)

    return pd.DataFrame(metric_rows), matrices


def threshold_experiment(
    y_train: pd.Series,
    probabilities: np.ndarray,
    thresholds: Iterable[float],
) -> pd.DataFrame:
    """Evaluate required thresholds on training out-of-fold scores."""

    rows = []

    for threshold in thresholds:
        y_pred = (
            probabilities >= threshold
        ).astype(int)

        rows.append(
            {
                "Threshold": float(threshold),
                **metric_dict(y_train, y_pred),
                **confusion_values(y_train, y_pred),
            }
        )

    return pd.DataFrame(rows)


def choose_model(
    results: dict[str, CVResult],
) -> str:
    """Choose the final candidate using CV only.

    Ranking:
    1. higher mean F1,
    2. higher mean Recall,
    3. smaller absolute generalization gap,
    4. lower F1 standard deviation.

    Business costs are not supplied by the assignment, so this reproducible
    rule balances detection, stability, and overfitting evidence.
    """

    def gap_penalty(result: CVResult) -> float:
        gap = result.generalization_gap
        return abs(gap) if gap is not None else float("inf")

    return max(
        results,
        key=lambda name: (
            results[name].mean_f1,
            results[name].mean_recall,
            -gap_penalty(results[name]),
            -results[name].f1_std,
        ),
    )


def choose_threshold(table: pd.DataFrame) -> float:
    """Choose a threshold from validation evidence, never the final test set."""

    ordered = table.sort_values(
        by=["F1", "Recall", "Precision"],
        ascending=[False, False, False],
        kind="mergesort",
    )

    return float(ordered.iloc[0]["Threshold"])


def fit_final_model(
    model_name: str,
    best_tree_depth: int | None,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> tuple[Any, StandardScaler | None]:
    """Fit the frozen final model on all training data."""

    if model_name == "PyTorch MLP":
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_train)

        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        model = train_torch_model(
            X_scaled,
            y_train,
            device=device,
        )
        return model, scaler

    if model_name == "Logistic Regression":
        scaler = StandardScaler()
        model = LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_STATE,
        )

    elif model_name == "KNN":
        scaler = StandardScaler()
        model = KNeighborsClassifier(
            n_neighbors=5,
            weights="uniform",
            n_jobs=-1,
        )

    elif model_name == "Decision Tree":
        scaler = None
        model = DecisionTreeClassifier(
            max_depth=best_tree_depth,
            random_state=RANDOM_STATE,
        )

    else:
        raise ValueError(f"Unsupported model: {model_name}")

    if scaler is not None:
        X_for_model = scaler.fit_transform(X_train)
    else:
        X_for_model = X_train

    model.fit(X_for_model, y_train)

    return model, scaler


def final_probabilities(
    model_name: str,
    model: Any,
    scaler: StandardScaler | None,
    X: pd.DataFrame,
) -> np.ndarray:
    """Return probabilities from the fitted final model."""

    if scaler is not None:
        X_for_model = scaler.transform(X)
    else:
        X_for_model = X

    if model_name == "PyTorch MLP":
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        model = model.to(device)
        return torch_probabilities(
            model,
            np.asarray(X_for_model, dtype=np.float32),
            device=device,
        )

    return sklearn_positive_probability(
        model,
        X_for_model,
    )


def make_model_artifact(
    model_name: str,
    model: Any,
    threshold: float,
    best_tree_depth: int | None,
) -> dict[str, Any]:
    """Create the persisted final model artifact."""

    common = {
        "artifact_type": "fraud_detection_model",
        "model_kind": (
            "pytorch" if model_name == "PyTorch MLP" else "sklearn"
        ),
        "model_name": model_name,
        "feature_columns": FEATURE_COLUMNS,
        "threshold": float(threshold),
        "positive_class": 1,
        "random_state": RANDOM_STATE,
        "best_tree_depth": best_tree_depth,
    }

    if model_name == "PyTorch MLP":
        return {
            **common,
            "model_config": copy.deepcopy(MLP_CONFIG),
            "model_state_dict": {
                key: tensor.detach().cpu()
                for key, tensor in model.state_dict().items()
            },
            "training_config": {
                "epochs": MLP_EPOCHS,
                "batch_size": MLP_BATCH_SIZE,
                "learning_rate": MLP_LEARNING_RATE,
                "optimizer": "Adam",
                "loss": "BCEWithLogitsLoss",
                "class_imbalance": (
                    "pos_weight = negative_count / positive_count"
                ),
            },
        }

    return {
        **common,
        "model": model,
    }


def markdown_table(
    df: pd.DataFrame,
    *,
    digits: int = 4,
) -> str:
    """Render a DataFrame as Markdown without an extra dependency."""

    display = df.copy()

    for column in display.columns:
        if pd.api.types.is_float_dtype(display[column]):
            display[column] = display[column].map(
                lambda value: (
                    "" if pd.isna(value) else f"{value:.{digits}f}"
                )
            )

    headers = [str(column) for column in display.columns]

    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]

    for _, row in display.iterrows():
        lines.append(
            "| " + " | ".join(str(value) for value in row.tolist()) + " |"
        )

    return "\n".join(lines)


def confusion_markdown(
    matrices: dict[str, dict[str, int]],
) -> str:
    """Render one confusion matrix summary per model."""

    sections = []

    for name, values in matrices.items():
        sections.append(
            f"""### {name}

|  | Predicted Legitimate | Predicted Fraud |
|---|---:|---:|
| Actual Legitimate | {values["TN"]} | {values["FP"]} |
| Actual Fraud | {values["FN"]} | {values["TP"]} |

- FP: {values["FP"]}
- FN: {values["FN"]}
"""
        )

    return "\n".join(sections)


def threshold_interpretation(
    threshold_table: pd.DataFrame,
    selected_threshold: float,
) -> str:
    """Create data-grounded threshold interpretation text."""

    ordered = threshold_table.sort_values("Threshold")
    low = ordered.iloc[0]
    high = ordered.iloc[-1]
    chosen = ordered.loc[
        np.isclose(ordered["Threshold"], selected_threshold)
    ].iloc[0]

    return f"""Lowering the threshold from {high["Threshold"]:.1f} to
{low["Threshold"]:.1f} changed Recall from {high["Recall"]:.4f} to
{low["Recall"]:.4f} and Precision from {high["Precision"]:.4f} to
{low["Precision"]:.4f}.

The selected threshold is **{selected_threshold:.1f}** under the project rule
of highest validation F1, then higher Recall, then higher Precision.

At this operating point, the training OOF confusion counts are:
TP={int(chosen["TP"])}, FP={int(chosen["FP"])},
FN={int(chosen["FN"])}, TN={int(chosen["TN"])}.

The practical trade-off is therefore explicit: reducing missed fraud (FN)
can increase false alarms (FP), while reducing false alarms can increase
missed fraud. No external financial cost matrix was supplied, so this project
uses validation metrics rather than claiming a business-optimal threshold.
"""


def tree_overfitting_analysis(
    depth_table: pd.DataFrame,
    best_tree_depth: int | None,
) -> str:
    """Create a data-grounded overfitting interpretation."""

    unrestricted = depth_table.loc[
        depth_table["max_depth"] == "None"
    ].iloc[0]

    selected_label = (
        "None" if best_tree_depth is None else str(best_tree_depth)
    )
    selected = depth_table.loc[
        depth_table["max_depth"] == selected_label
    ].iloc[0]

    gap_none = float(unrestricted["Generalization Gap"])
    gap_selected = float(selected["Generalization Gap"])

    if gap_none > gap_selected:
        comparison = (
            "The unrestricted tree shows a larger Train–Validation F1 gap "
            "than the selected depth, which is evidence of greater overfitting."
        )
    elif gap_none < gap_selected:
        comparison = (
            "The unrestricted tree does not show a larger gap than the selected "
            "depth in this run, so overfitting cannot be inferred from depth alone."
        )
    else:
        comparison = (
            "The unrestricted and selected trees show the same measured "
            "Train–Validation F1 gap in this run."
        )

    return f"""Selected Decision Tree depth: **{selected_label}**.

- Unrestricted tree gap: {gap_none:.4f}
- Selected tree gap: {gap_selected:.4f}

{comparison}

The selected value is determined by validation F1, Recall, stability, and
generalization evidence rather than training score alone.
"""


def update_readme_after_training(
    readme_path: Path,
    *,
    selected_model: str,
    selected_threshold: float,
    final_test_metrics: dict[str, float],
    final_test_confusion: dict[str, int],
    fraud_ratio: float,
    scaling_table: pd.DataFrame,
    tree_analysis: str,
    threshold_analysis: str,
) -> None:
    """Replace the README auto-results block with actual execution results."""

    text = readme_path.read_text(encoding="utf-8")

    start = "<!-- AUTO_RESULTS_START -->"
    end = "<!-- AUTO_RESULTS_END -->"

    if start not in text or end not in text:
        raise RuntimeError(
            "README auto-results markers are missing."
        )

    scaled = scaling_table.loc[
        scaling_table["Scaling"] == "With StandardScaler"
    ].iloc[0]
    unscaled = scaling_table.loc[
        scaling_table["Scaling"] == "Without Scaling"
    ].iloc[0]

    scaling_statement = (
        "supported"
        if float(scaled["F1"]) > float(unscaled["F1"])
        else "not supported by the observed F1 result"
    )

    replacement = f"""{start}
### Actual After-Training Analysis

This section was generated by `src/train.py` from the real dataset.

- Selected final model: **{selected_model}**
- Selected threshold: **{selected_threshold:.1f}**
- Final Test Accuracy: **{final_test_metrics["Accuracy"]:.4f}**
- Final Test Precision: **{final_test_metrics["Precision"]:.4f}**
- Final Test Recall: **{final_test_metrics["Recall"]:.4f}**
- Final Test F1: **{final_test_metrics["F1"]:.4f}**
- Final Test TP: **{final_test_confusion["TP"]}**
- Final Test TN: **{final_test_confusion["TN"]}**
- Final Test FP: **{final_test_confusion["FP"]}**
- Final Test FN: **{final_test_confusion["FN"]}**

#### Was the initial hypothesis correct?

The hypothesis that KNN would be meaningfully affected by scaling was
**{scaling_statement}** in this run:

- KNN F1 without scaling: {float(unscaled["F1"]):.4f}
- KNN F1 with scaling: {float(scaled["F1"]):.4f}

The Decision Tree overfitting hypothesis is evaluated from its
Train–Validation gap:

{tree_analysis}

#### Which model performed best?

Under the documented CV-only selection rule, **{selected_model}** was selected.
The final test set was not used to make this choice.

#### Which metric was most informative?

Because the fraud class is rare (fraud ratio = {fraud_ratio:.6f}), Accuracy
alone is not informative enough. The selection rule uses F1 as the primary
summary metric, with Recall, generalization gap, and CV stability as additional
evidence. Precision and the confusion matrix remain essential for interpreting
false alarms.

#### How did class imbalance affect the results?

The minority fraud class is much smaller than the legitimate class. Therefore,
the project reports Fraud Precision, Fraud Recall, F1, FP, and FN instead of
selecting a model from Accuracy alone.

#### What was the FP/FN trade-off?

{threshold_analysis}
{end}"""

    before = text.split(start, 1)[0]
    after = text.split(end, 1)[1]

    readme_path.write_text(
        before + replacement + after,
        encoding="utf-8",
    )


def write_report(
    report_path: Path,
    *,
    summary: dict[str, Any],
    structure: pd.DataFrame,
    descriptive: pd.DataFrame,
    cv_table: pd.DataFrame,
    oof_table: pd.DataFrame,
    oof_confusions: dict[str, dict[str, int]],
    scaling_table: pd.DataFrame,
    depth_table: pd.DataFrame,
    tree_analysis: str,
    threshold_table: pd.DataFrame,
    threshold_analysis: str,
    selected_model: str,
    selected_threshold: float,
    final_test_metrics: dict[str, float],
    final_test_confusion: dict[str, int],
) -> None:
    """Write the complete assignment report with real execution results."""

    report = f"""# Fraud Detection Experiments

> Generated by `src/train.py` from the real dataset.
> No project metric is hard-coded or fabricated.

## 1. Dataset Structure

- Number of samples: {summary["samples"]}
- Number of model features: {summary["model_features"]}
- Total columns including target: {summary["total_columns"]}
- Missing values: {summary["missing_values"]}
- Infinite feature values: {summary["infinite_feature_values"]}
- Duplicate rows observed: {summary["duplicate_rows"]}
- Legitimate transactions: {summary["legitimate_count"]}
- Fraudulent transactions: {summary["fraud_count"]}
- Fraud ratio: {summary["fraud_ratio"]:.6f}

### Column Structure

{markdown_table(structure)}

## 2. Descriptive Statistics

{markdown_table(descriptive)}

## 3. Model Evaluation — 5-Fold Stratified Cross Validation

### Mean Fold Metrics

{markdown_table(cv_table)}

### Aggregated Out-of-Fold Metrics at Threshold 0.5

This table reports all required metrics for every implemented model without
using the final test set for model comparison.

{markdown_table(oof_table)}

### Confusion Matrix for Every Model

{confusion_markdown(oof_confusions)}

Accuracy is reported but is not used as the primary selection metric because
the dataset is severely imbalanced.

## 4. Bonus PyTorch MLP

Architecture:

```text
30 → Linear(64) → ReLU → Dropout(0.20)
   → Linear(32) → ReLU → Dropout(0.20)
   → Linear(1 logit)
```

Training:

- Framework: PyTorch
- Loss: BCEWithLogitsLoss
- Class imbalance: `pos_weight = negative_count / positive_count`
- Optimizer: Adam
- Learning rate: {MLP_LEARNING_RATE}
- Epochs per fold: {MLP_EPOCHS}
- Batch size: {MLP_BATCH_SIZE}
- Scaling: StandardScaler fitted only on each fold's training partition
- Probability: sigmoid(logit)

## 5. Mandatory Experiment 1 — Effect of Scaling on KNN

{markdown_table(scaling_table)}

KNN is distance-based, so unequal numerical scales can dominate neighbor
distance. Decision Trees are much less sensitive to monotonic rescaling because
their decisions are based on ordered threshold splits.

## 6. Mandatory Experiment 2 — Decision Tree Hyperparameter Analysis

{markdown_table(depth_table)}

### Overfitting Analysis

{tree_analysis}

## 7. Mandatory Experiment 3 — Classification Threshold

{markdown_table(threshold_table)}

### Threshold Interpretation and Recommendation

{threshold_analysis}

## 8. Final Model Selection

Selected model: **{selected_model}**

Selected threshold: **{selected_threshold:.1f}**

The model was selected before the final test evaluation using:
1. Mean CV F1
2. Fraud Recall
3. Absolute Train–Validation generalization gap
4. F1 stability across folds

The threshold was selected from training OOF probabilities only.

This design considers class imbalance, overfitting evidence, Precision/Recall,
and the FP/FN trade-off while keeping the final test set independent.

## 9. Final Untouched Test Evaluation

The model, hyperparameters, preprocessing, and threshold were frozen first.
The test set was then evaluated exactly once.

- Accuracy: {final_test_metrics["Accuracy"]:.4f}
- Precision: {final_test_metrics["Precision"]:.4f}
- Recall: {final_test_metrics["Recall"]:.4f}
- F1-score: {final_test_metrics["F1"]:.4f}
- TP: {final_test_confusion["TP"]}
- TN: {final_test_confusion["TN"]}
- FP: {final_test_confusion["FP"]}
- FN: {final_test_confusion["FN"]}

## 10. Business Interpretation

- **False Negative:** fraudulent transaction predicted as legitimate; this is
  missed fraud and can create financial loss.
- **False Positive:** legitimate transaction predicted as fraud; this is a false
  alarm and can create customer friction and investigation cost.
- **Fraud Recall:** how much real fraud is detected.
- **Fraud Precision:** how trustworthy fraud alerts are.
- **F1:** balances Precision and Recall when no explicit cost matrix is supplied.
- **Accuracy:** reported for completeness but not used alone because the majority
  legitimate class can dominate it.

## 11. Required Question — Why Can Accuracy Be High but Fraud Detection Poor?

If a classifier predicts almost every transaction as legitimate, it can obtain
very high Accuracy because legitimate transactions dominate the dataset. At the
same time, Fraud Recall can be near zero because fraudulent transactions are
missed. Therefore Accuracy alone is not a reliable fraud-detection criterion.
"""

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    """Parse training command-line arguments."""

    root = Path(__file__).resolve().parent.parent

    parser = argparse.ArgumentParser(
        description=(
            "Run all mandatory experiments and the optional PyTorch MLP."
        )
    )

    parser.add_argument(
        "--data",
        type=Path,
        default=root / "data" / "creditcard.csv",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=root / "models",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=root / "reports" / "experiments.md",
    )
    parser.add_argument(
        "--readme",
        type=Path,
        default=root / "README.md",
    )

    return parser.parse_args()


def main() -> None:
    """Execute the complete project workflow."""

    seed_everything()
    args = parse_args()

    # ------------------------------------------------------------------
    # Phase 1 and Phase 2: data preparation, analysis, and protected split
    # ------------------------------------------------------------------
    df = load_dataset(args.data)

    summary_obj = dataset_summary(df)
    summary = summary_obj.as_dict()
    structure = dataset_structure(df)
    descriptive = descriptive_statistics(df)

    X, y = split_features_target(df)

    X_train, X_test, y_train, y_test = stratified_train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    print("Dataset summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")

    print("\nDescriptive statistics:")
    print(descriptive.to_string(index=False))

    cv = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    # ------------------------------------------------------------------
    # Phase 3: required models and bonus PyTorch model
    # ------------------------------------------------------------------
    results: dict[str, CVResult] = {}

    required_estimators = {
        "Logistic Regression": logistic_candidate(),
        "KNN": knn_candidate(scaled=True),
        "Decision Tree": tree_candidate(None),
    }

    for name, estimator in required_estimators.items():
        print(f"\nCross-validating {name}...")
        results[name] = evaluate_sklearn_cv(
            estimator,
            X_train,
            y_train,
            cv,
        )

    print("\nCross-validating bonus PyTorch MLP...")
    results["PyTorch MLP"] = evaluate_torch_cv(
        X_train,
        y_train,
        cv,
    )

    # ------------------------------------------------------------------
    # Mandatory Experiment 1: scaling
    # ------------------------------------------------------------------
    print("\nRunning mandatory KNN scaling experiment...")

    knn_unscaled = evaluate_sklearn_cv(
        knn_candidate(scaled=False),
        X_train,
        y_train,
        cv,
    )
    knn_scaled = results["KNN"]

    scaling_table = pd.DataFrame(
        [
            {
                "Model": "KNN",
                "Scaling": "Without Scaling",
                "Precision": knn_unscaled.mean_precision,
                "Recall": knn_unscaled.mean_recall,
                "F1": knn_unscaled.mean_f1,
                "F1 STD": knn_unscaled.f1_std,
            },
            {
                "Model": "KNN",
                "Scaling": "With StandardScaler",
                "Precision": knn_scaled.mean_precision,
                "Recall": knn_scaled.mean_recall,
                "F1": knn_scaled.mean_f1,
                "F1 STD": knn_scaled.f1_std,
            },
        ]
    )

    # ------------------------------------------------------------------
    # Mandatory Experiment 2: Decision Tree max_depth
    # ------------------------------------------------------------------
    print("\nRunning mandatory Decision Tree max_depth experiment...")

    depth_results: dict[int | None, CVResult] = {}
    depth_rows = []

    for depth in TREE_DEPTHS:
        result = evaluate_sklearn_cv(
            tree_candidate(depth),
            X_train,
            y_train,
            cv,
        )
        depth_results[depth] = result

        depth_rows.append(
            {
                "max_depth": "None" if depth is None else str(depth),
                "Train F1": result.mean_train_f1,
                "Validation Precision": result.mean_precision,
                "Validation Recall": result.mean_recall,
                "Validation F1": result.mean_f1,
                "F1 STD": result.f1_std,
                "Generalization Gap": result.generalization_gap,
            }
        )

    depth_table = pd.DataFrame(depth_rows)

    best_tree_depth = max(
        depth_results,
        key=lambda depth: (
            depth_results[depth].mean_f1,
            depth_results[depth].mean_recall,
            -abs(depth_results[depth].generalization_gap or 0.0),
            -depth_results[depth].f1_std,
        ),
    )

    # Use the tuned tree in the final candidate comparison.
    results["Decision Tree"] = depth_results[best_tree_depth]

    tree_analysis = tree_overfitting_analysis(
        depth_table,
        best_tree_depth,
    )

    # ------------------------------------------------------------------
    # Model evaluation for every model: all metrics + confusion matrix
    # ------------------------------------------------------------------
    cv_table = cv_mean_table(results)

    oof_table, oof_confusions = oof_model_evaluation(
        results,
        y_train,
    )

    # ------------------------------------------------------------------
    # Final model selection — CV only
    # ------------------------------------------------------------------
    selected_model = choose_model(results)

    # ------------------------------------------------------------------
    # Mandatory Experiment 3: thresholds using selected model OOF scores
    # ------------------------------------------------------------------
    threshold_table = threshold_experiment(
        y_train,
        results[selected_model].oof_probability,
        THRESHOLDS,
    )

    selected_threshold = choose_threshold(threshold_table)

    threshold_analysis = threshold_interpretation(
        threshold_table,
        selected_threshold,
    )

    # ------------------------------------------------------------------
    # Freeze configuration, fit all training data, then test once
    # ------------------------------------------------------------------
    final_model, final_scaler = fit_final_model(
        selected_model,
        best_tree_depth,
        X_train,
        y_train,
    )

    test_probability = final_probabilities(
        selected_model,
        final_model,
        final_scaler,
        X_test,
    )

    y_test_pred = (
        test_probability >= selected_threshold
    ).astype(int)

    final_test_metrics = metric_dict(
        y_test,
        y_test_pred,
    )
    final_test_confusion = confusion_values(
        y_test,
        y_test_pred,
    )

    # ------------------------------------------------------------------
    # Model saving
    # ------------------------------------------------------------------
    args.models_dir.mkdir(parents=True, exist_ok=True)

    model_path = args.models_dir / "model.pkl"
    scaler_path = args.models_dir / "scaler.pkl"
    encoder_path = args.models_dir / "encoder.pkl"

    artifact = make_model_artifact(
        selected_model,
        final_model,
        selected_threshold,
        best_tree_depth,
    )

    joblib.dump(artifact, model_path)
    joblib.dump(final_scaler, scaler_path)

    # All project features are numeric, so no categorical encoder is required.
    # The assignment names encoder.pkl in the Model Saving section; storing None
    # preserves that artifact contract without inventing preprocessing.
    joblib.dump(None, encoder_path)

    # ------------------------------------------------------------------
    # Prediction component smoke test using the actual saved artifacts
    # ------------------------------------------------------------------
    from predict import load_artifacts, predict_transaction

    loaded_artifact, loaded_scaler, loaded_encoder = load_artifacts(
        model_path,
        scaler_path,
        encoder_path,
    )

    sample_payload = {
        key: float(value)
        for key, value in X_test.iloc[0].to_dict().items()
    }

    smoke_result = predict_transaction(
        sample_payload,
        loaded_artifact,
        loaded_scaler,
        loaded_encoder,
    )

    if not (0.0 <= float(smoke_result["probability"]) <= 1.0):
        raise RuntimeError(
            "predict.py smoke test returned an invalid probability."
        )

    # ------------------------------------------------------------------
    # Complete report and README after-training analysis
    # ------------------------------------------------------------------
    write_report(
        args.report,
        summary=summary,
        structure=structure,
        descriptive=descriptive,
        cv_table=cv_table,
        oof_table=oof_table,
        oof_confusions=oof_confusions,
        scaling_table=scaling_table,
        depth_table=depth_table,
        tree_analysis=tree_analysis,
        threshold_table=threshold_table,
        threshold_analysis=threshold_analysis,
        selected_model=selected_model,
        selected_threshold=selected_threshold,
        final_test_metrics=final_test_metrics,
        final_test_confusion=final_test_confusion,
    )

    update_readme_after_training(
        args.readme,
        selected_model=selected_model,
        selected_threshold=selected_threshold,
        final_test_metrics=final_test_metrics,
        final_test_confusion=final_test_confusion,
        fraud_ratio=summary["fraud_ratio"],
        scaling_table=scaling_table,
        tree_analysis=tree_analysis,
        threshold_analysis=threshold_analysis,
    )

    print("\nTraining workflow completed successfully.")
    print(f"Selected model: {selected_model}")
    print(f"Selected threshold: {selected_threshold:.1f}")
    print(f"Final Test Accuracy: {final_test_metrics['Accuracy']:.4f}")
    print(f"Final Test Precision: {final_test_metrics['Precision']:.4f}")
    print(f"Final Test Recall: {final_test_metrics['Recall']:.4f}")
    print(f"Final Test F1: {final_test_metrics['F1']:.4f}")
    print(f"Model artifact: {model_path}")
    print(f"Scaler artifact: {scaler_path}")
    print(f"Encoder artifact: {encoder_path}")
    print(f"Report: {args.report}")
    print("predict.py smoke test: PASSED")


if __name__ == "__main__":
    main()
