"""
Deepfake Guardian 2026 — ML Calibration Engine

Trains a Random Forest classifier on the labeled dataset (dataset/real/ + dataset/fake/).
Extracts features from every video, trains the model, evaluates accuracy,
and saves the trained model to trained_model.pkl.

Usage:
    python calibration_engine.py
"""

import os
import sys
import json
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.pipeline import Pipeline

from feature_extractor import extract_features, features_to_vector, FEATURE_NAMES

# Paths
DATASET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset")
REAL_DIR = os.path.join(DATASET_DIR, "real")
FAKE_DIR = os.path.join(DATASET_DIR, "fake")
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trained_model.pkl")
FEATURES_CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "features_cache.json")


def scan_dataset():
    """Find all labeled videos in dataset/real and dataset/fake."""
    videos = []

    if os.path.isdir(REAL_DIR):
        for f in os.listdir(REAL_DIR):
            if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                videos.append({
                    "path": os.path.join(REAL_DIR, f),
                    "label": 0,  # 0 = real/authentic
                    "name": f,
                    "class": "REAL"
                })

    if os.path.isdir(FAKE_DIR):
        for f in os.listdir(FAKE_DIR):
            if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                videos.append({
                    "path": os.path.join(FAKE_DIR, f),
                    "label": 1,  # 1 = fake/synthetic
                    "name": f,
                    "class": "FAKE"
                })

    return videos


def load_cached_features():
    """Load previously extracted features from cache."""
    if os.path.exists(FEATURES_CACHE):
        try:
            with open(FEATURES_CACHE, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def save_cached_features(cache):
    """Save extracted features to cache."""
    with open(FEATURES_CACHE, "w") as f:
        json.dump(cache, f, indent=2)


def extract_dataset_features(videos, force_reextract=False):
    """
    Extract features from all videos in the dataset.
    Uses caching to avoid re-processing videos that haven't changed.
    """
    cache = load_cached_features() if not force_reextract else {}
    all_features = []
    all_labels = []

    for i, video in enumerate(videos):
        video_key = video["path"]
        video_mtime = str(os.path.getmtime(video["path"]))

        # Check cache
        if (video_key in cache and
            cache[video_key].get("mtime") == video_mtime and
            not force_reextract):
            print(f"  [{i+1}/{len(videos)}] Using cached features for: {video['name']} ({video['class']})")
            features = cache[video_key]["features"]
        else:
            print(f"  [{i+1}/{len(videos)}] Extracting features from: {video['name']} ({video['class']})")
            features = extract_features(video["path"])
            # Convert non-serializable types
            serializable_features = {}
            for k, v in features.items():
                if isinstance(v, (np.floating, np.integer)):
                    serializable_features[k] = float(v)
                elif isinstance(v, np.bool_):
                    serializable_features[k] = bool(v)
                else:
                    serializable_features[k] = v
            features = serializable_features
            cache[video_key] = {"features": features, "mtime": video_mtime}
            save_cached_features(cache)

        feature_vector = features_to_vector(features)
        all_features.append(feature_vector)
        all_labels.append(video["label"])

    return np.array(all_features), np.array(all_labels), cache


def train_model(X, y):
    """
    Train an ensemble classifier pipeline.
    Uses StandardScaler + GradientBoosting for best accuracy on small datasets.
    """
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', GradientBoostingClassifier(
            n_estimators=200,
            max_depth=3,
            learning_rate=0.1,
            min_samples_split=2,
            min_samples_leaf=1,
            subsample=0.8,
            random_state=42
        ))
    ])

    pipeline.fit(X, y)
    return pipeline


def evaluate_model(pipeline, X, y):
    """
    Evaluate using Leave-One-Out Cross-Validation (best for small datasets).
    Each video is tested with a model trained on all other videos.
    """
    loo = LeaveOneOut()
    predictions = cross_val_predict(pipeline, X, y, cv=loo)

    accuracy = accuracy_score(y, predictions)
    precision = precision_score(y, predictions, zero_division=0)
    recall = recall_score(y, predictions, zero_division=0)
    f1 = f1_score(y, predictions, zero_division=0)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "predictions": predictions.tolist(),
        "actual": y.tolist(),
        "report": classification_report(y, predictions, target_names=["REAL", "FAKE"], zero_division=0)
    }


def run_calibration(force_reextract=False, progress_callback=None):
    """
    Full calibration pipeline:
    1. Scan dataset
    2. Extract features from all videos
    3. Train the classifier
    4. Evaluate with Leave-One-Out CV
    5. Save the trained model
    """
    print("=" * 60)
    print("  DEEPFAKE GUARDIAN — MODEL CALIBRATION")
    print("=" * 60)

    # Step 1: Scan
    print("\n[1/4] Scanning dataset...")
    videos = scan_dataset()
    real_count = sum(1 for v in videos if v["label"] == 0)
    fake_count = sum(1 for v in videos if v["label"] == 1)
    print(f"  Found {len(videos)} videos: {real_count} REAL, {fake_count} FAKE")

    if len(videos) < 4:
        print("ERROR: Need at least 4 videos (2 real + 2 fake) for training.")
        return None

    # Step 2: Extract features
    print("\n[2/4] Extracting forensic features from all videos...")
    X, y, cache = extract_dataset_features(videos, force_reextract=force_reextract)
    print(f"  Feature matrix shape: {X.shape}")

    # Step 3: Train
    print("\n[3/4] Training Gradient Boosting classifier...")
    model = train_model(X, y)

    # Step 4: Evaluate
    print("\n[4/4] Evaluating with Leave-One-Out Cross-Validation...")
    metrics = evaluate_model(model, X, y)

    print(f"\n{'=' * 60}")
    print(f"  CALIBRATION RESULTS")
    print(f"{'=' * 60}")
    print(f"  Accuracy:  {metrics['accuracy']*100:.1f}%")
    print(f"  Precision: {metrics['precision']*100:.1f}%")
    print(f"  Recall:    {metrics['recall']*100:.1f}%")
    print(f"  F1 Score:  {metrics['f1_score']*100:.1f}%")
    print(f"\n{metrics['report']}")

    # Print per-video results
    print("\nPer-video predictions:")
    for i, video in enumerate(videos):
        actual = "REAL" if y[i] == 0 else "FAKE"
        predicted = "REAL" if metrics["predictions"][i] == 0 else "FAKE"
        status = "✅" if actual == predicted else "❌"
        print(f"  {status} {video['name']}: actual={actual}, predicted={predicted}")

    # Save model (trained on ALL data for production use)
    print(f"\nSaving trained model to {MODEL_PATH}...")
    joblib.dump(model, MODEL_PATH)

    # Save metadata
    meta = {
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1_score": metrics["f1_score"],
        "num_real": real_count,
        "num_fake": fake_count,
        "feature_names": FEATURE_NAMES,
    }
    meta_path = MODEL_PATH.replace(".pkl", "_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n✅ Calibration complete! Model saved.")
    return metrics


if __name__ == "__main__":
    force = "--force" in sys.argv
    run_calibration(force_reextract=force)
