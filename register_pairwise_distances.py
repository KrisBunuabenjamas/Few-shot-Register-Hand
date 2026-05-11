#!/usr/bin/env python3
"""
Register training dataset using pairwise distances between all 21 landmarks.
Creates 21x21 = 441 features (distance from each landmark to every other landmark).
Normalized by hand size for scale invariance.
"""

import os
import sys
import json
import argparse
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import numpy as np
import math


def get_pairwise_distances(hand_landmarks):
    """Legacy wrapper: accepts old-style hand_landmarks with .landmark list."""
    lm = hand_landmarks.landmark
    return _get_pairwise_distances_from_landmarks(lm)


def _get_pairwise_distances_from_landmarks(lm):
    """lm: list of landmark objects with .x and .y attributes (Tasks API or legacy)."""
    hand_size = math.sqrt((lm[12].x - lm[0].x)**2 + (lm[12].y - lm[0].y)**2) + 1e-6
    pairwise_dists = []
    for i in range(21):
        for j in range(21):
            dx = lm[i].x - lm[j].x
            dy = lm[i].y - lm[j].y
            pairwise_dists.append(math.sqrt(dx*dx + dy*dy) / hand_size)
    features = np.array(pairwise_dists)
    norm = np.linalg.norm(features)
    return features / norm if norm > 0 else features


def register_dataset_pairwise(dataset_dir, out_path, max_per_class=0, min_detection_confidence=0.5, min_score=0.7, prioritize_right=True):
    if not os.path.isdir(dataset_dir):
        raise FileNotFoundError(dataset_dir)

    _model_path = os.path.join(os.path.dirname(__file__), "hand_landmarker.task")
    options = mp_vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=_model_path),
        running_mode=mp_vision.RunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=min_detection_confidence,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    detector = mp_vision.HandLandmarker.create_from_options(options)

    all_samples = []
    stats = {}

    class_dirs = sorted([d for d in os.listdir(dataset_dir)
                         if os.path.isdir(os.path.join(dataset_dir, d))])

    for class_name in class_dirs:
        class_path = os.path.join(dataset_dir, class_name)
        image_files = sorted([f for f in os.listdir(class_path)
                               if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

        if max_per_class > 0:
            image_files = image_files[:max_per_class]

        detected_count = 0
        total_count = len(image_files)

        for img_file in image_files:
            img_path = os.path.join(class_path, img_file)
            image = cv2.imread(img_path)
            if image is None:
                continue

            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = detector.detect(mp_image)

            if not result.hand_landmarks:
                continue

            # Select best hand (prioritize right hand if multiple detected)
            best_idx = 0
            if len(result.handedness) > 1 and prioritize_right:
                for idx, handedness in enumerate(result.handedness):
                    if handedness[0].category_name == 'Right':
                        best_idx = idx
                        break

            hand_lm = result.hand_landmarks[best_idx]
            hand_label = result.handedness[best_idx][0].category_name
            hand_score = result.handedness[best_idx][0].score

            if hand_score < min_score:
                continue

            features = _get_pairwise_distances_from_landmarks(hand_lm)

            sample = {
                'class': class_name,
                'features': features.tolist(),
                'side': hand_label,
                'score': hand_score,
            }
            all_samples.append(sample)
            detected_count += 1

        stats[class_name] = {'total': total_count, 'detected': detected_count}

    detector.close()

    # Write to JSON
    output_data = {
        'samples': all_samples,
        'feature_type': 'pairwise_distances_21x21',
        'feature_dim': 441,
        'stats': stats
    }

    with open(out_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    return output_data, stats


def main():
    parser = argparse.ArgumentParser(description='Register hand posture dataset using pairwise distances (21x21)')
    parser.add_argument('--dataset', required=True, help='Path to dataset directory')
    parser.add_argument('--out', required=True, help='Output JSON file path')
    parser.add_argument('--max-per-class', type=int, default=0, help='Max images per class (0=all)')
    parser.add_argument('--min-confidence', type=float, default=0.5, help='Min detection confidence')
    parser.add_argument('--min-score', type=float, default=0.7, help='Min hand score')
    parser.add_argument('--prioritize-right', action='store_true', default=True, help='Prioritize right hand')

    args = parser.parse_args()

    output_data, stats = register_dataset_pairwise(
        args.dataset,
        args.out,
        max_per_class=args.max_per_class,
        min_detection_confidence=args.min_confidence,
        min_score=args.min_score,
        prioritize_right=args.prioritize_right
    )

    print(f"Registered {len(output_data['samples'])} training samples with pairwise distances")
    print(f"   Feature type: 21x21 pairwise distances (441 features)")
    print()
    print("Stats by class:")
    for cls, st in stats.items():
        print(f"  {cls}: total={st['total']}, detected={st['detected']}")
    print()
    print(f"Wrote: {args.out}")


if __name__ == '__main__':
    main()
