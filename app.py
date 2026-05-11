#!/usr/bin/env python3
"""Flask web app for ASL hand gesture recognition."""

import os
import json
import math
import threading
import time
import types
from collections import Counter
from io import BytesIO

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from flask import Flask, Response, jsonify, render_template, request

app = Flask(__name__)

# Global state
state = {
    "training_data": None,
    "training_path": None,
    "camera_active": False,
    "latest_prediction": {"class": None, "confidence": 0.0, "votes": []},
    "k": 3,
    "metric": "euclidean",
    "latest_landmarks": None,
    "pending_samples": [],
}
state_lock = threading.Lock()

# MediaPipe Tasks API
_MODEL_PATH = os.path.join(os.path.dirname(__file__), "hand_landmarker.task")
_hand_options = mp_vision.HandLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path=_MODEL_PATH),
    running_mode=mp_vision.RunningMode.IMAGE,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)
hands_detector = mp_vision.HandLandmarker.create_from_options(_hand_options)

# Hand connection pairs for drawing
_HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),(0,17),
]


# ── Feature extraction ────────────────────────────────────────────────────────

def get_pairwise_distances(lm):
    """lm: list of NormalizedLandmark objects with .x, .y"""
    hand_size = math.sqrt((lm[12].x - lm[0].x) ** 2 + (lm[12].y - lm[0].y) ** 2) + 1e-6
    dists = []
    for i in range(21):
        for j in range(21):
            dx = lm[i].x - lm[j].x
            dy = lm[i].y - lm[j].y
            dists.append(math.sqrt(dx * dx + dy * dy) / hand_size)
    features = np.array(dists)
    norm = np.linalg.norm(features)
    return features / norm if norm > 0 else features


def distance_metric(v1, v2, metric):
    if metric == "euclidean":
        return np.linalg.norm(v1 - v2)
    if metric == "manhattan":
        return np.sum(np.abs(v1 - v2))
    if metric == "cosine":
        return 1.0 - np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
    if metric == "mse":
        return np.mean((v1 - v2) ** 2)
    return np.linalg.norm(v1 - v2)


def knn_classify(test_features, samples, k, metric):
    dists = []
    for s in samples:
        d = distance_metric(test_features, np.array(s["features"]), metric)
        dists.append((d, s["class"]))
    dists.sort(key=lambda x: x[0])
    votes = [c for _, c in dists[:k]]
    counts = Counter(votes)
    best = counts.most_common(1)[0][0]
    best_dist = min(d for d, c in dists[:k] if c == best)
    confidence = 1.0 / (best_dist + 1e-6)
    return best, confidence, votes


# ── Video streaming ───────────────────────────────────────────────────────────

def generate_frames():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return

    with state_lock:
        state["camera_active"] = True

    try:
        while True:
            with state_lock:
                if not state["camera_active"]:
                    break

            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            detection = hands_detector.detect(mp_image)

            pred_class = None
            confidence = 0.0
            votes = []

            if detection.hand_landmarks:
                lm = detection.hand_landmarks[0]
                with state_lock:
                    state["latest_landmarks"] = [(p.x, p.y, p.z) for p in lm]
                h, w = frame.shape[:2]

                # Draw landmarks
                pts = [(int(p.x * w), int(p.y * h)) for p in lm]
                for a, b in _HAND_CONNECTIONS:
                    cv2.line(frame, pts[a], pts[b], (0, 200, 255), 2)
                for pt in pts:
                    cv2.circle(frame, pt, 4, (255, 255, 255), -1)

                with state_lock:
                    training = state["training_data"]
                    k = state["k"]
                    metric = state["metric"]

                if training:
                    features = get_pairwise_distances(lm)
                    pred_class, confidence, votes = knn_classify(
                        features, training["samples"], k, metric
                    )

                    cv2.putText(
                        frame, f"{pred_class}  {confidence:.1f}",
                        (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5,
                        (0, 255, 0), 3, cv2.LINE_AA,
                    )

            if not detection.hand_landmarks:
                with state_lock:
                    state["latest_landmarks"] = None

            with state_lock:
                state["latest_prediction"] = {
                    "class": pred_class,
                    "confidence": round(float(confidence), 2),
                    "votes": votes,
                }

            _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")
    finally:
        cap.release()
        with state_lock:
            state["camera_active"] = False


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/prediction")
def prediction():
    with state_lock:
        return jsonify(state["latest_prediction"])


@app.route("/status")
def status():
    with state_lock:
        return jsonify({
            "training_loaded": state["training_data"] is not None,
            "training_path": state["training_path"],
            "training_samples": len(state["training_data"]["samples"]) if state["training_data"] else 0,
            "camera_active": state["camera_active"],
            "k": state["k"],
            "metric": state["metric"],
        })


@app.route("/load_training", methods=["POST"])
def load_training():
    data = request.get_json()
    path = data.get("path", "").strip()
    if not path or not os.path.isfile(path):
        return jsonify({"ok": False, "error": "File not found"}), 400
    try:
        with open(path) as f:
            td = json.load(f)
        with state_lock:
            state["training_data"] = td
            state["training_path"] = path
        return jsonify({"ok": True, "samples": len(td["samples"])})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400


@app.route("/upload_training", methods=["POST"])
def upload_training():
    if "file" not in request.files:
        return jsonify({"ok": False, "error": "No file"}), 400
    f = request.files["file"]
    try:
        td = json.load(f)
        with state_lock:
            state["training_data"] = td
            state["training_path"] = f.filename
        return jsonify({"ok": True, "samples": len(td["samples"])})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400


@app.route("/settings", methods=["POST"])
def settings():
    data = request.get_json()
    with state_lock:
        if "k" in data:
            state["k"] = int(data["k"])
        if "metric" in data:
            state["metric"] = data["metric"]
    return jsonify({"ok": True})


@app.route("/stop_camera", methods=["POST"])
def stop_camera():
    with state_lock:
        state["camera_active"] = False
    return jsonify({"ok": True})


@app.route("/class_stats")
def class_stats():
    """Load and return class statistics from evaluation_results.json if it exists."""
    results_file = os.path.join(os.path.dirname(__file__), "evaluation_results.json")
    if not os.path.isfile(results_file):
        return jsonify({"ok": False, "error": "No evaluation_results.json found"})
    try:
        with open(results_file) as f:
            data = json.load(f)
        return jsonify({"ok": True, "data": data})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})


@app.route("/capture_sample", methods=["POST"])
def capture_sample():
    data = request.get_json()
    class_name = (data.get("class_name") or "").strip()
    if not class_name:
        return jsonify({"ok": False, "error": "class_name required"}), 400
    with state_lock:
        lm_raw = state["latest_landmarks"]
    if lm_raw is None:
        return jsonify({"ok": False, "error": "No hand detected"}), 400
    lm = [types.SimpleNamespace(x=t[0], y=t[1]) for t in lm_raw]
    features = get_pairwise_distances(lm)
    sample = {"class": class_name, "features": features.tolist(), "side": "Unknown", "score": 1.0}
    with state_lock:
        state["pending_samples"].append(sample)
        count = len(state["pending_samples"])
    return jsonify({"ok": True, "count": count})


@app.route("/commit_gesture", methods=["POST"])
def commit_gesture():
    data = request.get_json()
    class_name = (data.get("class_name") or "").strip()
    if not class_name:
        return jsonify({"ok": False, "error": "class_name required"}), 400
    with state_lock:
        if state["training_data"] is None:
            return jsonify({"ok": False, "error": "No training data loaded"}), 400
        added = [s for s in state["pending_samples"] if s["class"] == class_name]
        state["training_data"]["samples"].extend(added)
        stats = state["training_data"].setdefault("stats", {})
        if class_name in stats:
            stats[class_name]["total"] += len(added)
            stats[class_name]["detected"] = stats[class_name].get("detected", 0) + len(added)
        else:
            stats[class_name] = {"total": len(added), "detected": len(added)}
        state["pending_samples"] = []
        total = len(state["training_data"]["samples"])
    return jsonify({"ok": True, "added": len(added), "total_samples": total})


@app.route("/save_training", methods=["POST"])
def save_training():
    with state_lock:
        td = state["training_data"]
        path = state["training_path"]
    if td is None or not path:
        return jsonify({"ok": False, "error": "No training data loaded"}), 400
    try:
        with open(path, "w") as f:
            json.dump(td, f, indent=2)
        return jsonify({"ok": True, "path": path})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/clear_pending", methods=["POST"])
def clear_pending():
    with state_lock:
        state["pending_samples"] = []
    return jsonify({"ok": True})


def _autoload_training():
    default = os.path.join(os.path.dirname(__file__), "asl_training.json")
    if os.path.isfile(default):
        try:
            with open(default) as f:
                td = json.load(f)
            with state_lock:
                state["training_data"] = td
                state["training_path"] = default
            print(f"Auto-loaded {len(td['samples'])} samples from {default}")
        except Exception as e:
            print(f"Could not auto-load training data: {e}")


if __name__ == "__main__":
    _autoload_training()
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
