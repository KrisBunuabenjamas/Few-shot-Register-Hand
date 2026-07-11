"""Feature extraction helpers for one-hand and two-hand gesture samples."""

import math

import numpy as np


HAND_LANDMARK_COUNT = 21
PAIRWISE_FEATURE_DIM = HAND_LANDMARK_COUNT * HAND_LANDMARK_COUNT
TWO_HAND_FEATURE_DIM = PAIRWISE_FEATURE_DIM * 3
TWO_HAND_FEATURE_TYPE = "pairwise_distances_2hands_21x21_cross"


def as_landmark_list(hand_landmarks):
    """Support both MediaPipe Solutions landmarks and Tasks landmark lists."""
    if hasattr(hand_landmarks, "landmark"):
        return list(hand_landmarks.landmark)
    return list(hand_landmarks)


def hand_size(landmarks):
    return math.sqrt(
        (landmarks[12].x - landmarks[0].x) ** 2
        + (landmarks[12].y - landmarks[0].y) ** 2
    ) + 1e-6


def hand_center_x(landmarks):
    return sum(p.x for p in landmarks) / len(landmarks)


def get_pairwise_distances(hand_landmarks):
    """Return the original 21x21 normalized pairwise-distance feature vector."""
    lm = as_landmark_list(hand_landmarks)
    scale = hand_size(lm)
    dists = []
    for i in range(HAND_LANDMARK_COUNT):
        for j in range(HAND_LANDMARK_COUNT):
            dx = lm[i].x - lm[j].x
            dy = lm[i].y - lm[j].y
            dists.append(math.sqrt(dx * dx + dy * dy) / scale)

    features = np.array(dists)
    norm = np.linalg.norm(features)
    return features / norm if norm > 0 else features


def normalize_hands(detected_hands):
    """Place detected hands into stable left/right slots."""
    slots = {"Left": None, "Right": None}
    leftovers = []

    for hand in detected_hands:
        landmarks = as_landmark_list(hand["landmarks"])
        item = {
            "landmarks": landmarks,
            "label": hand.get("label"),
            "score": hand.get("score", 0.0),
            "center_x": hand_center_x(landmarks),
        }
        label = item["label"]
        if label in slots and slots[label] is None:
            slots[label] = item
        else:
            leftovers.append(item)

    for item in sorted(leftovers, key=lambda h: h["center_x"]):
        if slots["Left"] is None:
            slots["Left"] = item
        elif slots["Right"] is None:
            slots["Right"] = item

    return slots


def get_cross_hand_distances(left_landmarks, right_landmarks):
    """Return distances between every left-hand point and every right-hand point."""
    left_size = hand_size(left_landmarks)
    right_size = hand_size(right_landmarks)
    scale = (left_size + right_size) / 2.0

    dists = []
    for i in range(HAND_LANDMARK_COUNT):
        for j in range(HAND_LANDMARK_COUNT):
            dx = left_landmarks[i].x - right_landmarks[j].x
            dy = left_landmarks[i].y - right_landmarks[j].y
            dists.append(math.sqrt(dx * dx + dy * dy) / scale)

    features = np.array(dists)
    norm = np.linalg.norm(features)
    return features / norm if norm > 0 else features


def get_two_hand_pairwise_features(detected_hands):
    """
    Return a fixed-size 1323-dim vector:
    left hand 441 + right hand 441 + cross-hand 441.
    Missing hands are represented by zero-filled chunks.
    """
    slots = normalize_hands(detected_hands)
    zero_chunk = np.zeros(PAIRWISE_FEATURE_DIM)

    left = slots["Left"]
    right = slots["Right"]
    left_features = get_pairwise_distances(left["landmarks"]) if left else zero_chunk
    right_features = get_pairwise_distances(right["landmarks"]) if right else zero_chunk
    cross_features = (
        get_cross_hand_distances(left["landmarks"], right["landmarks"])
        if left and right
        else zero_chunk
    )

    features = np.concatenate([left_features, right_features, cross_features])
    norm = np.linalg.norm(features)
    return features / norm if norm > 0 else features
