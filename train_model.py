"""Train the 3-class direction classifier and report honest out-of-sample metrics.

Reads train_data.csv (from build_training_data.py), holds out the newest 20% of bars, reports
accuracy vs. the majority baseline and the precision of directional calls at several confidence
thresholds, picks a recommended threshold, refits on everything, and saves a bundle to model.pkl.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from bot.features import FEATURE_COLUMNS, LABEL_DOWN, LABEL_FLAT, LABEL_NAMES, LABEL_UP

TRAIN_CSV = os.getenv("TRAIN_DATA_CSV", "train_data.csv")
MODEL_PATH = os.getenv("MODEL_PATH", "model.pkl")
THRESHOLDS = (0.40, 0.45, 0.50, 0.55, 0.60, 0.65)
MIN_COVERAGE = 0.02  # a threshold must still act on >= 2% of bars to be useful
MIN_EDGE = 0.05  # right-direction rate must exceed wrong-direction rate by this much


def build_model(model_type: str):
    if model_type == "xgboost":
        from xgboost import XGBClassifier

        return XGBClassifier(
            n_estimators=400,
            max_depth=4,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=5,
            reg_lambda=2.0,
            random_state=42,
            eval_metric="mlogloss",
            n_jobs=-1,
        )
    return RandomForestClassifier(
        n_estimators=400,
        max_depth=8,
        min_samples_leaf=20,
        max_features="sqrt",
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )


def directional_report(probs: np.ndarray, classes: np.ndarray, y_true: np.ndarray) -> dict:
    """Precision of up/down calls at each confidence threshold; returns the recommended one."""
    idx = {int(c): i for i, c in enumerate(classes)}
    p_up = probs[:, idx[LABEL_UP]] if LABEL_UP in idx else np.zeros(len(probs))
    p_down = probs[:, idx[LABEL_DOWN]] if LABEL_DOWN in idx else np.zeros(len(probs))

    # Baseline: if you call a direction at random, how often is it right?
    moved = y_true != LABEL_FLAT
    baseline = 0.5 * moved.mean() if moved.any() else 0.0

    print(f"\nDirectional calls on held-out bars (random-direction baseline {baseline:.3f}):")
    print("  edge = right-way minus wrong-way; flat outcomes are neither")
    print(f"  {'thr':>5} {'coverage':>9} {'calls':>6} {'right-way':>10} {'wrong-way':>10} {'edge':>7}")
    recommended = None
    for thr in THRESHOLDS:
        call_up = (p_up >= thr) & (p_up >= p_down)
        call_down = (p_down >= thr) & (p_down > p_up)
        calls = call_up | call_down
        n = int(calls.sum())
        coverage = n / len(y_true)
        if n == 0:
            print(f"  {thr:>5.2f} {coverage:>9.1%} {n:>6} {'-':>10} {'-':>10} {'-':>7}")
            continue
        correct = (call_up & (y_true == LABEL_UP)) | (call_down & (y_true == LABEL_DOWN))
        wrong = (call_up & (y_true == LABEL_DOWN)) | (call_down & (y_true == LABEL_UP))
        precision = correct.sum() / n
        wrong_way = wrong.sum() / n
        edge = precision - wrong_way
        flag = ""
        if recommended is None and coverage >= MIN_COVERAGE and edge >= MIN_EDGE:
            recommended = thr
            flag = "  <- recommended"
        print(
            f"  {thr:>5.2f} {coverage:>9.1%} {n:>6} {precision:>10.3f} {wrong_way:>10.3f} {edge:>+7.3f}{flag}"
        )
    return {"baseline": float(baseline), "recommended": recommended}


def main() -> None:
    df = pd.read_csv(TRAIN_CSV)
    missing = [c for c in FEATURE_COLUMNS + ["target"] if c not in df.columns]
    if missing:
        raise SystemExit(f"{TRAIN_CSV} is missing {missing}; run build_training_data.py first.")
    if "timestamp" in df.columns:
        df = df.sort_values("timestamp").reset_index(drop=True)

    X = df[FEATURE_COLUMNS].astype(float)
    y = df["target"].astype(int)
    model_type = os.getenv("ML_MODEL_TYPE", "xgboost").strip().lower()
    print(f"Rows: {len(X)}  features: {len(FEATURE_COLUMNS)}  model: {model_type}")
    mix = y.value_counts(normalize=True).sort_index()
    print("Class mix: " + ", ".join(f"{LABEL_NAMES[int(k)]}={v:.1%}" for k, v in mix.items()))

    report = {"baseline": None, "recommended": None}
    test_accuracy = None
    majority = None
    if len(X) >= 500:
        split = int(len(X) * 0.8)
        model = build_model(model_type)
        model.fit(X.iloc[:split], y.iloc[:split])
        test_X, test_y = X.iloc[split:], y.iloc[split:].to_numpy()
        preds = model.predict(test_X)
        test_accuracy = float((preds == test_y).mean())
        majority = float(pd.Series(test_y).value_counts(normalize=True).iloc[0])
        print(f"\nOut-of-sample 3-class accuracy: {test_accuracy:.3f} on {len(test_y)} bars")
        print(f"Always-majority baseline:       {majority:.3f}")
        report = directional_report(model.predict_proba(test_X), model.classes_, test_y)
        if report["recommended"] is None:
            print(
                "\nWARNING: no confidence threshold beats the baseline with usable coverage. "
                "The model has no measurable edge yet; keep it on paper."
            )
    else:
        print(f"WARNING: only {len(X)} rows; too few to evaluate. Run build_training_data.py first.")

    final_model = build_model(model_type)
    final_model.fit(X, y)
    recommended = report["recommended"] if report["recommended"] is not None else 0.60
    bundle = {
        "model": final_model,
        "model_type": model_type,
        "feature_columns": FEATURE_COLUMNS,
        "n_classes": int(len(np.unique(y))),
        "recommended_min_confidence": float(recommended),
        "has_edge": report["recommended"] is not None,
        "test_accuracy": test_accuracy,
        "majority_baseline": majority,
        "rows": int(len(X)),
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }
    joblib.dump(bundle, MODEL_PATH)
    print(
        f"\nSaved {MODEL_PATH}: {model_type}, {len(X)} rows, "
        f"recommended MIN_SIGNAL_CONFIDENCE={recommended:.2f}"
        + ("" if bundle["has_edge"] else " (default; no edge found)")
    )


if __name__ == "__main__":
    main()
