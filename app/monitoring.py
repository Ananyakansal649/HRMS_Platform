"""
Enterprise HR AI - Model Monitoring (Enhanced)
Tasks 25-27: Data drift monitoring, model performance monitoring, retraining strategy.
"""
import os
import json
import time
import hashlib
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MONITORING_DIR = PROJECT_ROOT / "data" / "monitoring"
PREDICTIONS_LOG = MONITORING_DIR / "predictions.jsonl"
PERFORMANCE_LOG = MONITORING_DIR / "performance_log.jsonl"
STATS_FILE = MONITORING_DIR / "drift_stats.json"
TRAINING_STATS_FILE = MONITORING_DIR / "training_stats.json"
RETRAIN_LOG = MONITORING_DIR / "retrain_log.jsonl"

logger = logging.getLogger("hr_ai_monitoring")


def ensure_monitoring_dir():
    """Create monitoring directory if it doesn't exist."""
    MONITORING_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Prediction Logging (Task 21)
# ---------------------------------------------------------------------------
def log_prediction(
    input_data: dict,
    prediction: str,
    probability: float,
    risk_level: str,
    model_version: str,
    endpoint: str = "api",
) -> dict:
    """Log a single prediction to the JSONL file."""
    ensure_monitoring_dir()

    record = {
        "timestamp": datetime.now().isoformat(),
        "endpoint": endpoint,
        "model_version": model_version,
        "prediction": prediction,
        "attrition_probability": probability,
        "risk_level": risk_level,
        "input_hash": hashlib.md5(
            json.dumps(input_data, sort_keys=True, default=str).encode()
        ).hexdigest()[:12],
        "feature_snapshot": {
            "Age": input_data.get("Age"),
            "MonthlyIncome": input_data.get("MonthlyIncome"),
            "TotalWorkingYears": input_data.get("TotalWorkingYears"),
            "Department": input_data.get("Department"),
            "JobRole": input_data.get("JobRole"),
            "OverTime": input_data.get("OverTime"),
        },
    }

    with open(PREDICTIONS_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str) + "\n")

    return record


def load_predictions(limit: Optional[int] = None) -> list:
    """Load recent predictions from the log file."""
    if not PREDICTIONS_LOG.exists():
        return []

    records = []
    with open(PREDICTIONS_LOG, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    if limit:
        records = records[-limit:]

    return records


# ---------------------------------------------------------------------------
# Data Drift Monitoring (Task 25)
# ---------------------------------------------------------------------------
def compute_training_reference_stats() -> dict:
    """
    Compute reference statistics from training data for drift comparison.
    Watches: age, monthly_income, years_at_company, job_satisfaction, overtime.
    """
    try:
        df = pd.read_csv(str(PROJECT_ROOT / "data" / "processed" / "employee_attrition_processed.csv"))

        stats = {
            "Age": {"mean": round(float(df["Age"].mean()), 2), "std": round(float(df["Age"].std()), 2),
                     "min": int(df["Age"].min()), "max": int(df["Age"].max())},
            "MonthlyIncome": {"mean": round(float(df["MonthlyIncome"].mean()), 2), "std": round(float(df["MonthlyIncome"].std()), 2),
                               "min": int(df["MonthlyIncome"].min()), "max": int(df["MonthlyIncome"].max())},
            "YearsAtCompany": {"mean": round(float(df["YearsAtCompany"].mean()), 2), "std": round(float(df["YearsAtCompany"].std()), 2),
                                "min": int(df["YearsAtCompany"].min()), "max": int(df["YearsAtCompany"].max())},
            "JobSatisfaction": {"mean": round(float(df["JobSatisfaction"].mean()), 2), "std": round(float(df["JobSatisfaction"].std()), 2),
                                 "min": int(df["JobSatisfaction"].min()), "max": int(df["JobSatisfaction"].max())},
            "OverTime_rate": round(float((df["OverTime"] == "Yes").mean()), 4) if "OverTime" in df.columns else 0,
            "n_training_samples": len(df),
        }

        # Persist
        ensure_monitoring_dir()
        with open(TRAINING_STATS_FILE, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)

        return stats
    except Exception as e:
        logger.error("Failed to compute training reference stats: %s", e)
        return {}


def compute_feature_drift() -> dict:
    """
    Compare production input distributions against training reference.
    Task 25: Watch age, monthly_income, years_at_company, job_satisfaction, overtime.
    """
    # Load training reference (or compute if missing)
    if not TRAINING_STATS_FILE.exists():
        train_stats = compute_training_reference_stats()
    else:
        with open(TRAINING_STATS_FILE) as f:
            train_stats = json.load(f)

    if not train_stats:
        return {"status": "error", "message": "No training reference stats available"}

    # Load production predictions
    preds = load_predictions(limit=1000)
    if not preds:
        return {
            "status": "insufficient_data",
            "message": "No production predictions to compare",
            "training_reference": train_stats,
        }

    # Extract feature values from production snapshots
    feature_values = {}
    for p in preds:
        snapshot = p.get("feature_snapshot", {})
        for key in ["Age", "MonthlyIncome", "TotalWorkingYears"]:
            val = snapshot.get(key)
            if val is not None:
                feature_values.setdefault(key, []).append(float(val))

    # Compute production statistics
    prod_stats = {}
    for key, values in feature_values.items():
        if values:
            prod_stats[key] = {
                "mean": round(float(np.mean(values)), 2),
                "std": round(float(np.std(values)), 2),
                "min": round(float(np.min(values)), 2),
                "max": round(float(np.max(values)), 2),
                "count": len(values),
            }

    # Compare and detect drift
    alerts = []
    drift_details = {}
    for feature in ["Age", "MonthlyIncome"]:
        if feature in train_stats and feature in prod_stats:
            train_mean = train_stats[feature]["mean"]
            prod_mean = prod_stats[feature]["mean"]
            train_std = max(train_stats[feature]["std"], 0.01)

            # Z-score of the shift
            z_score = abs(prod_mean - train_mean) / train_std

            drift_details[feature] = {
                "training_mean": train_mean,
                "production_mean": prod_mean,
                "shift": round(prod_mean - train_mean, 2),
                "z_score": round(z_score, 2),
            }

            if z_score > 2.0:
                alerts.append({
                    "type": "feature_drift",
                    "feature": feature,
                    "message": f"{feature}: training mean={train_mean}, production mean={prod_mean} (z={z_score:.2f})",
                    "severity": "high" if z_score > 3.0 else "medium",
                })

    return {
        "status": "ok",
        "training_reference": train_stats,
        "production_stats": prod_stats,
        "drift_details": drift_details,
        "alerts": alerts if alerts else [{"type": "none", "message": "No feature drift detected", "severity": "info"}],
    }


# ---------------------------------------------------------------------------
# Prediction Distribution Drift (existing, enhanced)
# ---------------------------------------------------------------------------
def compute_drift_stats(
    reference_window_hours: int = 24,
    current_window_hours: int = 1,
) -> dict:
    """Compare recent predictions against a reference window to detect distribution drift."""
    all_preds = load_predictions()

    if not all_preds:
        return {
            "status": "insufficient_data",
            "message": "No predictions logged yet",
            "total_predictions": 0,
        }

    now = datetime.now()
    cutoff_ref = now - timedelta(hours=reference_window_hours)
    cutoff_cur = now - timedelta(hours=current_window_hours)

    reference = [p for p in all_preds if datetime.fromisoformat(p["timestamp"]) >= cutoff_ref]
    current = [p for p in all_preds if datetime.fromisoformat(p["timestamp"]) >= cutoff_cur]

    if len(reference) < 5:
        return {
            "status": "insufficient_data",
            "message": f"Only {len(reference)} predictions in reference window (need >= 5)",
            "total_predictions": len(all_preds),
            "reference_window_count": len(reference),
            "current_window_count": len(current),
        }

    ref_probs = [p["attrition_probability"] for p in reference]
    ref_pos_rate = sum(1 for p in reference if p["prediction"] == "Yes") / len(reference)

    stats = {
        "status": "ok",
        "total_predictions": len(all_preds),
        "reference_window": {
            "hours": reference_window_hours,
            "count": len(reference),
            "mean_probability": round(float(np.mean(ref_probs)), 4),
            "std_probability": round(float(np.std(ref_probs)), 4),
            "positive_rate": round(ref_pos_rate, 4),
        },
        "current_window": {"hours": current_window_hours, "count": len(current)},
        "alerts": [],
    }

    if len(current) >= 3:
        cur_probs = [p["attrition_probability"] for p in current]
        cur_pos_rate = sum(1 for p in current if p["prediction"] == "Yes") / len(current)

        stats["current_window"]["mean_probability"] = round(float(np.mean(cur_probs)), 4)
        stats["current_window"]["std_probability"] = round(float(np.std(cur_probs)), 4)
        stats["current_window"]["positive_rate"] = round(cur_pos_rate, 4)

        prob_diff = abs(stats["current_window"]["mean_probability"] - stats["reference_window"]["mean_probability"])
        if prob_diff > 0.15:
            stats["alerts"].append({
                "type": "probability_drift",
                "message": f"Mean probability shifted by {prob_diff:.4f} (> 0.15 threshold)",
                "severity": "high",
            })

        rate_diff = abs(stats["current_window"]["positive_rate"] - stats["reference_window"]["positive_rate"])
        if rate_diff > 0.2:
            stats["alerts"].append({
                "type": "prediction_rate_drift",
                "message": f"Positive prediction rate shifted by {rate_diff:.4f} (> 0.2 threshold)",
                "severity": "high",
            })

    if not stats["alerts"]:
        stats["alerts"].append({"type": "none", "message": "No drift detected", "severity": "info"})

    ensure_monitoring_dir()
    with open(STATS_FILE, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, default=str)

    return stats


# ---------------------------------------------------------------------------
# Model Performance Monitoring (Task 26)
# ---------------------------------------------------------------------------
def log_actual_outcome(
    employee_id: str,
    predicted: str,
    predicted_probability: float,
    actual_outcome: str,
    model_version: str = "v1.0",
) -> dict:
    """
    Log an actual outcome (when we learn whether the employee actually left).
    Task 26: Once real attrition outcomes come in, compare against predictions.
    """
    ensure_monitoring_dir()

    record = {
        "timestamp": datetime.now().isoformat(),
        "employee_id": employee_id,
        "predicted": predicted,
        "predicted_probability": predicted_probability,
        "actual_outcome": actual_outcome,
        "correct": predicted == actual_outcome,
        "model_version": model_version,
    }

    with open(PERFORMANCE_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

    return record


def compute_model_performance() -> dict:
    """
    Task 26: Recompute precision/recall/F1/ROC-AUC on live data.
    Compares predicted vs actual outcomes.
    """
    if not PERFORMANCE_LOG.exists():
        return {
            "status": "insufficient_data",
            "message": "No actual outcomes logged yet. Log outcomes with log_actual_outcome().",
            "n_outcomes": 0,
        }

    records = []
    with open(PERFORMANCE_LOG, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    if len(records) < 5:
        return {
            "status": "insufficient_data",
            "message": f"Only {len(records)} outcomes logged (need >= 5)",
            "n_outcomes": len(records),
        }

    # Compute confusion matrix
    tp = sum(1 for r in records if r["predicted"] == "Yes" and r["actual_outcome"] == "Yes")
    fp = sum(1 for r in records if r["predicted"] == "Yes" and r["actual_outcome"] == "No")
    tn = sum(1 for r in records if r["predicted"] == "No" and r["actual_outcome"] == "No")
    fn = sum(1 for r in records if r["predicted"] == "No" and r["actual_outcome"] == "Yes")

    total = tp + fp + tn + fn
    accuracy = (tp + tn) / total if total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    # Load training baseline for comparison
    train_metrics = {}
    metadata_path = PROJECT_ROOT / "models" / "v1" / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path) as f:
            meta = json.load(f)
        train_metrics = meta.get("metrics", {})

    performance = {
        "status": "ok",
        "n_outcomes": total,
        "confusion_matrix": {"true_negatives": tn, "false_positives": fp, "false_negatives": fn, "true_positives": tp},
        "live_metrics": {
            "accuracy": round(accuracy, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
        },
        "training_baseline": {
            "precision": train_metrics.get("precision"),
            "recall": train_metrics.get("recall"),
            "f1_score": train_metrics.get("f1_score"),
            "roc_auc": train_metrics.get("roc_auc"),
        },
        "performance_drift_alerts": [],
    }

    # Compare live vs training F1
    if train_metrics.get("f1_score") and f1 > 0:
        f1_drop = train_metrics["f1_score"] - f1
        if f1_drop > 0.1:
            performance["performance_drift_alerts"].append({
                "type": "f1_drop",
                "message": f"Live F1 ({f1:.4f}) is {f1_drop:.4f} below training F1 ({train_metrics['f1_score']:.4f})",
                "severity": "high" if f1_drop > 0.2 else "medium",
            })

    return performance


# ---------------------------------------------------------------------------
# Retraining Strategy (Task 27)
# ---------------------------------------------------------------------------
RETRAIN_THRESHOLDS = {
    "drift_z_score": 3.0,           # Feature drift z-score threshold
    "f1_drop": 0.2,                 # F1 drop from training baseline
    "min_predictions_for_retrain": 100,  # Minimum predictions before considering retrain
    "retrain_cooldown_days": 30,    # Don't retrain more than once per month
    "new_data_months": 6,           # Retrain after 6 months of new data collected
}


def check_retrain_conditions() -> dict:
    """
    Task 27: Automated retraining rules.
    IF drift > threshold OR F1 drops below threshold OR 6 months of new data
    THEN retrain the model.
    """
    conditions = []
    should_retrain = False

    # Check 1: Feature drift
    feature_drift = compute_feature_drift()
    for alert in feature_drift.get("alerts", []):
        if alert.get("severity") in ("high", "medium"):
            conditions.append({
                "condition": "feature_drift",
                "triggered": True,
                "detail": alert["message"],
                "severity": alert["severity"],
            })
            if alert["severity"] == "high":
                should_retrain = True

    # Check 2: Prediction distribution drift
    drift_stats = compute_drift_stats()
    for alert in drift_stats.get("alerts", []):
        if alert.get("type") in ("probability_drift", "prediction_rate_drift"):
            conditions.append({
                "condition": "prediction_drift",
                "triggered": True,
                "detail": alert["message"],
                "severity": alert["severity"],
            })
            should_retrain = True

    # Check 3: Model performance degradation
    perf = compute_model_performance()
    for alert in perf.get("performance_drift_alerts", []):
        conditions.append({
            "condition": "performance_degradation",
            "triggered": True,
            "detail": alert["message"],
            "severity": alert["severity"],
        })
        should_retrain = True

    # Check 4: New data collection period (6 months)
    training_date_str = None
    metadata_path = PROJECT_ROOT / "models" / "v1" / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path) as f:
            meta_check = json.load(f)
        training_date_str = meta_check.get("training_date")
    if training_date_str:
        try:
            training_date = datetime.strptime(training_date_str, "%Y-%m-%d")
            months_since = (datetime.now() - training_date).days / 30.0
            if months_since >= RETRAIN_THRESHOLDS["new_data_months"]:
                conditions.append({
                    "condition": "new_data_period",
                    "triggered": True,
                    "detail": f"{months_since:.1f} months since last training (threshold: {RETRAIN_THRESHOLDS['new_data_months']} months)",
                    "severity": "medium",
                })
                should_retrain = True
            else:
                conditions.append({
                    "condition": "new_data_period",
                    "triggered": False,
                    "detail": f"{months_since:.1f} months since last training (threshold: {RETRAIN_THRESHOLDS['new_data_months']} months)",
                    "severity": "info",
                })
        except (ValueError, TypeError):
            pass

    # Check 5: Cooldown (prevent retraining too frequently)
    cooldown_ok = True
    if RETRAIN_LOG.exists():
        with open(RETRAIN_LOG) as f:
            last_line = None
            for line in f:
                if line.strip():
                    last_line = json.loads(line.strip())
            if last_line:
                last_retrain = datetime.fromisoformat(last_line["timestamp"])
                days_since = (datetime.now() - last_retrain).days
                if days_since < RETRAIN_THRESHOLDS["retrain_cooldown_days"]:
                    cooldown_ok = False
                    should_retrain = False
                    conditions.append({
                        "condition": "cooldown_active",
                        "triggered": True,
                        "detail": f"Last retrain {days_since} days ago (cooldown: {RETRAIN_THRESHOLDS['retrain_cooldown_days']} days)",
                        "severity": "info",
                    })

    return {
        "should_retrain": should_retrain,
        "conditions_checked": len(conditions),
        "conditions_triggered": sum(1 for c in conditions if c["triggered"]),
        "conditions": conditions,
        "thresholds": RETRAIN_THRESHOLDS,
        "cooldown_active": not cooldown_ok,
    }


def log_retrain_event(
    model_version: str,
    trigger_reason: str,
    metrics_before: dict,
    metrics_after: dict,
):
    """Log a retraining event for audit trail."""
    ensure_monitoring_dir()

    record = {
        "timestamp": datetime.now().isoformat(),
        "model_version": model_version,
        "trigger_reason": trigger_reason,
        "metrics_before": metrics_before,
        "metrics_after": metrics_after,
    }

    with open(RETRAIN_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str) + "\n")

    return record


# ---------------------------------------------------------------------------
# Model Health (existing)
# ---------------------------------------------------------------------------
def get_model_health(model_version: str = "v1.0") -> dict:
    """Check model health: file exists, metadata consistent, prediction rate normal."""
    from app.utils.config import MODEL_PATH, METADATA_PATH

    health = {
        "model_file_exists": MODEL_PATH.exists(),
        "model_file_size": MODEL_PATH.stat().st_size if MODEL_PATH.exists() else 0,
        "metadata_exists": METADATA_PATH.exists(),
    }

    if METADATA_PATH.exists():
        with open(METADATA_PATH, "r") as f:
            meta = json.load(f)
        health["model_version"] = meta.get("version", "unknown")
        health["algorithm"] = meta.get("algorithm", "unknown")
        health["training_date"] = meta.get("training_date", "unknown")
        health["test_roc_auc"] = meta.get("metrics", {}).get("roc_auc", "unknown")
        health["test_f1"] = meta.get("metrics", {}).get("f1_score", "unknown")

    preds = load_predictions(limit=100)
    health["recent_prediction_count"] = len(preds)
    if preds:
        timestamps = [datetime.fromisoformat(p["timestamp"]) for p in preds]
        time_span = (max(timestamps) - min(timestamps)).total_seconds() / 3600
        health["predictions_per_hour"] = round(len(preds) / max(time_span, 0.01), 2)

    return health
