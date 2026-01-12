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
import numpy as np
import math


def get_pairwise_distances(hand_landmarks):
    """
    Calculate pairwise distances between all 21 hand landmarks.
    Returns a flattened array of 21x21 = 441 normalized distances.
    """
    lm = hand_landmarks.landmark
    
    # Calculate hand size for normalization (wrist to middle finger tip)
    hand_size = math.sqrt((lm[12].x - lm[0].x)**2 + (lm[12].y - lm[0].y)**2) + 1e-6
    
    # Calculate distance from each landmark to every other landmark
    pairwise_dists = []
    for i in range(21):
        for j in range(21):
            dx = lm[i].x - lm[j].x
            dy = lm[i].y - lm[j].y
            dist = math.sqrt(dx*dx + dy*dy) / hand_size
            pairwise_dists.append(dist)
    
    features = np.array(pairwise_dists)
    
    # Apply L2 normalization
    norm = np.linalg.norm(features)
    if norm > 0:
        features = features / norm
    
    return features


def register_dataset_pairwise(dataset_dir, out_path, max_per_class=0, min_detection_confidence=0.5, min_score=0.7, prioritize_right=True):
    if not os.path.isdir(dataset_dir):
        raise FileNotFoundError(dataset_dir)

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(max_num_hands=2,
                           min_detection_confidence=min_detection_confidence,
                           min_tracking_confidence=0.5)

    all_samples = []
    stats = {}

    # Walk through all subdirectories (each is a class)
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
            results = hands.process(rgb)

            if not results.multi_hand_landmarks:
                continue

            # Select best hand (prioritize right hand if multiple detected)
            best_idx = 0
            if len(results.multi_hand_landmarks) > 1 and prioritize_right:
                if results.multi_handedness[1].classification[0].label == 'Right':
                    best_idx = 1

            hand_lm = results.multi_hand_landmarks[best_idx]
            hand_label = results.multi_handedness[best_idx].classification[0].label
            hand_score = results.multi_handedness[best_idx].classification[0].score

            if hand_score < min_score:
                continue

            # Extract pairwise distance features
            features = get_pairwise_distances(hand_lm)

            sample = {
                'class': class_name,
                'features': features.tolist(),
                'side': hand_label,
                'score': hand_score
            }
            all_samples.append(sample)
            detected_count += 1

        stats[class_name] = {
            'total': total_count,
            'detected': detected_count
        }

    hands.close()

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

    print(f"✅ Registered {len(output_data['samples'])} training samples with pairwise distances")
    print(f"   Feature type: 21x21 pairwise distances (441 features)")
    print()
    print("Stats by class:")
    for cls, st in stats.items():
        print(f"  {cls}: total={st['total']}, detected={st['detected']}")
    print()
    print(f"Wrote: {args.out}")


if __name__ == '__main__':
    main()
