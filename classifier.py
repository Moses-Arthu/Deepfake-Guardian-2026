"""
Deepfake Guardian 2026 — ML Classifier

Loads the trained model and makes predictions on new videos.
Falls back to an advanced heuristic if no trained model exists.
Provides per-feature contribution breakdown for explainability.
"""

import os
import json
import numpy as np
import joblib

from feature_extractor import extract_features, features_to_vector, FEATURE_NAMES

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trained_model.pkl")
META_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trained_model_meta.json")


def load_model():
    """Load the trained model and metadata."""
    if not os.path.exists(MODEL_PATH):
        return None, None
    try:
        model = joblib.load(MODEL_PATH)
        meta = None
        if os.path.exists(META_PATH):
            with open(META_PATH) as f:
                meta = json.load(f)
        return model, meta
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None


def get_feature_contributions(model, feature_vector):
    """
    Get per-feature contribution to the prediction.
    Uses feature importances from the trained model.
    """
    try:
        # Get feature importances from the classifier in the pipeline
        classifier = model.named_steps['classifier']
        importances = classifier.feature_importances_

        contributions = {}
        for i, name in enumerate(FEATURE_NAMES):
            contributions[name] = {
                "importance": float(importances[i]),
                "value": float(feature_vector[i]),
                "rank": 0  # filled below
            }

        # Rank by importance
        sorted_features = sorted(contributions.items(), key=lambda x: x[1]["importance"], reverse=True)
        for rank, (name, info) in enumerate(sorted_features, 1):
            contributions[name]["rank"] = rank

        return contributions
    except Exception:
        return {}


def heuristic_prediction(features):
    """
    Advanced heuristic fallback when no trained model exists.
    Uses multiple weighted signals with more refined thresholds.
    """
    score = 0.0
    max_score = 0.0

    # --- Blink rate analysis (weight: 15%) ---
    bpm = features.get("blinks_per_minute", 15)
    if 8 <= bpm <= 35:
        blink_score = 0.0   # Normal range
    elif 3 <= bpm < 8 or 35 < bpm <= 60:
        blink_score = 0.3   # Slightly off
    elif bpm == 0:
        blink_score = 0.7   # No blinks detected
    else:
        blink_score = 0.8   # Very abnormal
    score += blink_score * 0.15
    max_score += 0.15

    # --- EAR variance (weight: 10%) ---
    ear_std = features.get("ear_std", 0.03)
    if ear_std < 0.008:
        ear_score = 0.9   # Unnaturally stable
    elif ear_std < 0.015:
        ear_score = 0.5   # Somewhat flat
    else:
        ear_score = 0.0   # Natural
    score += ear_score * 0.10
    max_score += 0.10

    # --- LBP texture uniformity (weight: 15%) ---
    lbp_uni = features.get("lbp_uniformity_mean", 0.05)
    if lbp_uni > 0.12:
        lbp_score = 0.9   # Too smooth/uniform
    elif lbp_uni > 0.08:
        lbp_score = 0.4   # Borderline
    else:
        lbp_score = 0.0   # Natural texture
    score += lbp_score * 0.15
    max_score += 0.15

    # --- DCT frequency analysis (weight: 15%) ---
    dct_hf = features.get("dct_high_freq_ratio", 0.01)
    if dct_hf < 0.003:
        dct_score = 0.8   # Suspiciously low high-freq
    elif dct_hf > 0.05:
        dct_score = 0.6   # Unusual high-freq artifacts
    else:
        dct_score = 0.0   # Normal
    score += dct_score * 0.15
    max_score += 0.15

    # --- Landmark jitter (weight: 10%) ---
    jitter = features.get("landmark_jitter", 1.0)
    if jitter > 5.0:
        jitter_score = 0.8   # Very jittery
    elif jitter < 0.2:
        jitter_score = 0.6   # Too stable (AI)
    else:
        jitter_score = 0.0   # Normal
    score += jitter_score * 0.10
    max_score += 0.10

    # --- Optical flow (weight: 10%) ---
    flow_std = features.get("optical_flow_std", 1.0)
    if flow_std < 0.3:
        flow_score = 0.6   # Too consistent motion
    else:
        flow_score = 0.0
    score += flow_score * 0.10
    max_score += 0.10

    # --- Audio features (weight: 15%) ---
    if features.get("has_audio", False):
        mfcc_var = features.get("mfcc_variance", 100)
        if mfcc_var < 20:
            audio_score = 0.7  # Very uniform speech
        else:
            audio_score = 0.0
        score += audio_score * 0.15
        max_score += 0.15

    # --- Lip sync (weight: 10%) ---
    lip_std = features.get("lip_distance_std", 0.02)
    if lip_std < 0.005:
        lip_score = 0.7   # Barely moving lips
    else:
        lip_score = 0.0
    score += lip_score * 0.10
    max_score += 0.10

    confidence = score / max_score if max_score > 0 else 0.0
    is_fake = confidence > 0.5

    return {
        "prediction": "SYNTHETIC" if is_fake else "AUTHENTIC",
        "confidence": confidence,
        "method": "heuristic_fallback",
        "contributions": {}
    }


def predict_video(video_path, progress_callback=None):
    """
    Make a deepfake prediction on a video file.

    Returns:
        dict with:
        - prediction: "AUTHENTIC" or "SYNTHETIC"
        - confidence: 0.0-1.0 (how confident the model is)
        - features: dict of all extracted features
        - contributions: per-feature importance breakdown
        - method: "ml_model" or "heuristic_fallback"
    """
    # Extract features
    features = extract_features(video_path, progress_callback=progress_callback)
    feature_vector = features_to_vector(features)

    # Try ML model first
    model, meta = load_model()

    if model is not None:
        # ML prediction
        prediction_label = model.predict(feature_vector.reshape(1, -1))[0]
        probabilities = model.predict_proba(feature_vector.reshape(1, -1))[0]

        # probability[0] = real, probability[1] = fake
        fake_prob = probabilities[1]
        real_prob = probabilities[0]

        prediction = "SYNTHETIC" if prediction_label == 1 else "AUTHENTIC"
        confidence = fake_prob if prediction_label == 1 else real_prob

        contributions = get_feature_contributions(model, feature_vector)

        result = {
            "prediction": prediction,
            "confidence": float(confidence),
            "fake_probability": float(fake_prob),
            "real_probability": float(real_prob),
            "features": features,
            "contributions": contributions,
            "method": "ml_model",
            "model_accuracy": meta.get("accuracy", 0) if meta else 0,
        }
    else:
        # Fallback to heuristic
        heuristic = heuristic_prediction(features)
        result = {
            "prediction": heuristic["prediction"],
            "confidence": heuristic["confidence"],
            "fake_probability": heuristic["confidence"] if heuristic["prediction"] == "SYNTHETIC" else 1 - heuristic["confidence"],
            "real_probability": 1 - heuristic["confidence"] if heuristic["prediction"] == "SYNTHETIC" else heuristic["confidence"],
            "features": features,
            "contributions": heuristic["contributions"],
            "method": "heuristic_fallback",
            "model_accuracy": 0,
        }

    return result


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python classifier.py <video_path>")
        sys.exit(1)

    video_path = sys.argv[1]
    print(f"Analyzing: {video_path}")
    result = predict_video(video_path)

    print(f"\n{'='*50}")
    print(f"  PREDICTION: {result['prediction']}")
    print(f"  Confidence: {result['confidence']*100:.1f}%")
    print(f"  Method: {result['method']}")
    print(f"{'='*50}")

    if result["contributions"]:
        print("\nTop contributing features:")
        sorted_contribs = sorted(
            result["contributions"].items(),
            key=lambda x: x[1]["importance"],
            reverse=True
        )[:5]
        for name, info in sorted_contribs:
            print(f"  {info['rank']}. {name}: importance={info['importance']:.3f}, value={info['value']:.4f}")
