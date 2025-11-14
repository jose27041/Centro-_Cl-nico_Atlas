from __future__ import annotations

import base64
import io
import json
import os
from pathlib import Path
from typing import Dict, List

import joblib
import matplotlib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.metrics import ConfusionMatrixDisplay  # noqa: E402

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
METRICS_PATH = MODELS_DIR / "metrics.json"

MODEL_LABELS = {
    "logistic_regression": "Regresion Logistica",
    "mlp_classifier": "Red Neuronal (MLP)",
}

CLASS_LABEL_NAME_MAP: Dict[int, str] = {
    1: "Dengue",
    2: "Malaria",
    3: "Leptospirosis",
}

def _ensure_numpy_bit_generator_patch() -> None:
    """
    Handles joblib artifacts created with newer NumPy versions where the
    bit-generator is serialized as a class instead of a string identifier.
    """
    try:
        import numpy.random._pickle as np_pickle  # type: ignore[attr-defined]
    except Exception:
        return

    original_ctor = getattr(np_pickle, "__bit_generator_ctor", None)
    if original_ctor is None or getattr(original_ctor, "__patched__", False):
        return

    def _patched(bit_generator_name="MT19937"):
        if not isinstance(bit_generator_name, str):
            for name, cls in np_pickle.BitGenerators.items():  # type: ignore[attr-defined]
                if cls is bit_generator_name or getattr(bit_generator_name, "__name__", None) == name:
                    bit_generator_name = name
                    break
            else:
                bit_generator_name = getattr(bit_generator_name, "__name__", str(bit_generator_name))
        return original_ctor(bit_generator_name)

    setattr(_patched, "__patched__", True)
    np_pickle.__bit_generator_ctor = _patched


_ensure_numpy_bit_generator_patch()


def load_assets() -> Dict[str, object]:
    if not METRICS_PATH.exists():
        raise FileNotFoundError(
            "No se encontro el archivo de metricas. Ejecuta 'python train_models.py' antes."
        )

    with METRICS_PATH.open("r", encoding="utf-8") as f:
        metrics_doc = json.load(f)

    feature_order: List[str] = metrics_doc.get("feature_order", [])
    class_labels: List[int] = metrics_doc.get("class_labels", [])
    class_distribution: Dict[str, Dict[str, int]] = metrics_doc.get("class_distribution", {})

    models: Dict[str, object] = {}
    model_metrics: Dict[str, dict] = {}

    for model_key in MODEL_LABELS:
        model_path = MODELS_DIR / f"{model_key}.joblib"
        if not model_path.exists():
            raise FileNotFoundError(f"No se encontro el modelo entrenado en {model_path}")
        models[model_key] = joblib.load(model_path)
        model_metrics[model_key] = metrics_doc["models"].get(model_key, {})

    return {
        "feature_order": feature_order,
        "models": models,
        "model_metrics": model_metrics,
        "class_labels": class_labels,
        "dataset": metrics_doc.get("dataset"),
        "dataset_description": metrics_doc.get("dataset_description"),
        "target_column": metrics_doc.get("target_column"),
        "class_distribution": class_distribution,
    }


ASSETS = load_assets()
TARGET_COL = ASSETS.get("target_column", "outcome")

app = Flask(__name__)
app.secret_key = os.environ.get("SCOUT_SECRET_KEY", "scout-demo-secret")
app.config.update(
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
)


def get_class_label_name(label: object) -> str:
    try:
        label_int = int(label)
    except (TypeError, ValueError):
        return str(label)
    return CLASS_LABEL_NAME_MAP.get(label_int, str(label_int))


def build_class_info(labels: List[object]) -> List[Dict[str, object]]:
    class_info: List[Dict[str, object]] = []
    for raw_label in labels:
        try:
            label_id = int(raw_label)
        except (TypeError, ValueError):
            label_id = raw_label
        class_info.append(
            {
                "id": label_id,
                "name": get_class_label_name(label_id),
            }
        )
    return class_info


def summarize_class_distribution(
    labels: List[object], distribution: Dict[str, Dict[str, int]]
) -> List[Dict[str, object]]:
    original = distribution.get("original") or {}
    balanced = distribution.get("balanced") or {}
    train_raw = distribution.get("train_before_balance") or {}
    train_balanced = distribution.get("train_balanced") or balanced
    test_counts = distribution.get("test") or {}

    def resolve_count(mapping: Dict[object, int], label_id: int) -> int:
        if label_id in mapping:
            return mapping[label_id]
        str_key = str(label_id)
        return mapping.get(str_key, 0)

    summary: List[Dict[str, object]] = []
    for info in build_class_info(labels):
        label_id = info["id"]
        summary.append(
            {
                **info,
                "original": resolve_count(original, label_id),
                "balanced": resolve_count(train_balanced, label_id),
                "train_before_balance": resolve_count(train_raw, label_id),
                "train_balanced": resolve_count(train_balanced, label_id),
                "test": resolve_count(test_counts, label_id),
            }
        )
    return summary


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/status")
def api_status():
    class_label_values = ASSETS.get("class_labels", [])
    class_distribution = ASSETS.get("class_distribution") or {}
    payload = {
        "feature_order": ASSETS["feature_order"],
        "models": [
            {"id": key, "label": MODEL_LABELS[key], "metrics": ASSETS["model_metrics"][key]}
            for key in MODEL_LABELS
        ],
        "class_labels": build_class_info(class_label_values),
        "dataset": {
            "id": ASSETS.get("dataset"),
            "description": ASSETS.get("dataset_description"),
            "target_column": TARGET_COL,
            "class_distribution": {
                **class_distribution,
                "summary": summarize_class_distribution(class_label_values, class_distribution),
            },
        },
    }
    return jsonify(payload)


def parse_features(payload: dict) -> pd.DataFrame:
    if "features" not in payload:
        raise ValueError("Falta el campo 'features' en la solicitud.")

    feature_order: List[str] = ASSETS["feature_order"]
    features = payload["features"]
    missing = [f for f in feature_order if f not in features]
    if missing:
        raise ValueError(
            f"Faltan variables requeridas: {', '.join(missing)}"
        )

    try:
        row = {key: float(features[key]) for key in feature_order}
    except (TypeError, ValueError) as exc:
        raise ValueError("Todos los valores deben ser numericos.") from exc

    negative_fields = [name for name, value in row.items() if value < 0]
    if negative_fields:
        raise ValueError(
            f"No se permiten valores negativos en: {', '.join(negative_fields)}"
        )

    return pd.DataFrame([row])


def get_model(model_key: str):
    if model_key not in MODEL_LABELS:
        raise KeyError(f"El modelo '{model_key}' no esta disponible.")
    return ASSETS["models"][model_key]


@app.post("/api/predict/individual")
def predict_individual():
    payload = request.get_json(silent=True)
    if payload is None:
        return jsonify({"error": "Cuerpo JSON invalido."}), 400

    model_key = payload.get("model", "logistic_regression")
    try:
        model = get_model(model_key)
        features_df = parse_features(payload)
    except (KeyError, ValueError) as error:
        return jsonify({"error": str(error)}), 400

    prediction = model.predict(features_df)[0]
    try:
        prediction_id = int(prediction)
    except (TypeError, ValueError):
        prediction_id = prediction

    class_raw_labels = list(map(int, getattr(model, "classes_", ASSETS.get("class_labels", []))))
    class_info = build_class_info(class_raw_labels)
    probability_payload = None
    if hasattr(model, "predict_proba"):
        proba_values = model.predict_proba(features_df)[0]
        probability_payload = [
            {
                "id": info["id"],
                "label": info["name"],
                "probability": float(prob),
            }
            for info, prob in zip(class_info, proba_values)
        ]

    response = {
        "model": {
            "id": model_key,
            "label": MODEL_LABELS.get(model_key, model_key),
        },
        "prediction": prediction_id,
        "prediction_label": get_class_label_name(prediction_id),
        "probabilities": probability_payload,
        "class_labels": class_info,
    }
    return jsonify(response)


def load_uploaded_dataframe(file_storage) -> pd.DataFrame:
    filename = file_storage.filename or ""
    suffix = filename.lower().split(".")[-1]
    buffer = io.BytesIO(file_storage.read())
    buffer.seek(0)
    if suffix == "csv":
        df = pd.read_csv(buffer)
    elif suffix in {"xlsx", "xls"}:
        df = pd.read_excel(buffer)
    else:
        raise ValueError("Formato de archivo no soportado. Usa archivos .csv o .xlsx.")
    return df


def normalize_confusion_matrix(cm: np.ndarray) -> np.ndarray:
    normalized = cm.astype(float)
    row_sums = normalized.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return normalized / row_sums


def generate_confusion_matrix_image(cm, labels: List[str], normalized: bool = False) -> str:
    fig, ax = plt.subplots(figsize=(4, 4))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    values_format = ".2f" if normalized else "d"
    disp.plot(
        cmap="Blues",
        ax=ax,
        colorbar=False,
        values_format=values_format,
    )
    title = "Matriz de confusion (normalizada)" if normalized else "Matriz de confusion (conteos)"
    ax.set_title(title)
    plt.tight_layout()
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=150)
    plt.close(fig)
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("ascii")


@app.post("/api/predict/batch")
def predict_batch():
    if "file" not in request.files:
        return jsonify({"error": "No se encontro el archivo enviado (campo 'file')."}), 400
    model_key = request.form.get("model", "logistic_regression")
    try:
        model = get_model(model_key)
    except KeyError as error:
        return jsonify({"error": str(error)}), 400

    uploaded_file = request.files["file"]

    try:
        df = load_uploaded_dataframe(uploaded_file)
    except ValueError as error:
        return jsonify({"error": str(error)}), 400

    feature_order: List[str] = ASSETS["feature_order"]

    missing_features = [f for f in feature_order if f not in df.columns]
    if missing_features:
        return (
            jsonify(
                {
                    "error": "Faltan columnas requeridas en el archivo.",
                    "missing_features": missing_features,
                }
            ),
            400,
        )

    if TARGET_COL not in df.columns:
        return (
            jsonify(
                {
                    "error": f"El archivo debe incluir la columna objetivo '{TARGET_COL}'.",
                }
            ),
            400,
        )

    try:
        X = df[feature_order].apply(pd.to_numeric, errors="raise")
    except ValueError as error:
        return jsonify({"error": f"No se pudieron convertir los valores a numericos: {error}"}), 400

    negative_columns = [col for col in feature_order if (X[col] < 0).any()]
    if negative_columns:
        return (
            jsonify(
                {
                    "error": "Se detectaron valores negativos en columnas no permitidas.",
                    "columnas": negative_columns,
                }
            ),
            400,
        )

    y_true = df[TARGET_COL].astype(int)

    predictions = model.predict(X)

    model_labels = list(map(int, getattr(model, "classes_", sorted(y_true.unique()))))
    class_info = build_class_info(model_labels)
    display_labels = [item["name"] for item in class_info]
    cm = confusion_matrix(y_true, predictions, labels=model_labels)
    cm_normalized = normalize_confusion_matrix(cm)
    metrics = {
        "accuracy": accuracy_score(y_true, predictions),
        "precision": precision_score(
            y_true, predictions, zero_division=0, average="weighted"
        ),
        "recall": recall_score(
            y_true, predictions, zero_division=0, average="weighted"
        ),
        "f1": f1_score(y_true, predictions, zero_division=0, average="weighted"),
    }
    report = classification_report(
        y_true, predictions, output_dict=True, zero_division=0
    )
    cm_image_counts = generate_confusion_matrix_image(cm, labels=display_labels)
    cm_image_normalized = generate_confusion_matrix_image(
        cm_normalized, labels=display_labels, normalized=True
    )

    named_predictions = [get_class_label_name(label) for label in predictions]
    preview = df.assign(prediction=named_predictions).head(10)

    response = {
        "model": {
            "id": model_key,
            "label": MODEL_LABELS.get(model_key, model_key),
        },
        "metrics": metrics,
        "classification_report": report,
        "confusion_matrix": {
            "labels": class_info,
            "matrix": cm.tolist(),
            "matrix_normalized": cm_normalized.tolist(),
            "image_png_base64": cm_image_counts,
            "image_counts_png_base64": cm_image_counts,
            "image_normalized_png_base64": cm_image_normalized,
        },
        "preview": preview.to_dict(orient="records"),
        "total_samples": int(len(df)),
        "class_labels": class_info,
    }
    return jsonify(response)


if __name__ == "__main__":
    app.run(debug=True)
