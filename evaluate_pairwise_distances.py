#!/usr/bin/env python3
"""
Evaluate hand posture recognition using pairwise distances between all 21 landmarks.
Uses KNN classifier with 21x21 = 441 features.
"""

import os
import sys
import json
import argparse
import cv2
import mediapipe as mp
import numpy as np
import math
import time
from collections import Counter
from PIL import Image


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


def apply_zoom(image, zoom_factor):
    """Apply zoom to image (zoom_factor > 1.0 zooms in, < 1.0 zooms out)"""
    if abs(zoom_factor - 1.0) < 0.01:
        return image.copy()
    
    h, w = image.shape[:2]
    
    if zoom_factor > 1.0:  # Zoom in
        new_h, new_w = int(h * zoom_factor), int(w * zoom_factor)
        resized = cv2.resize(image, (new_w, new_h))
        start_y = (new_h - h) // 2
        start_x = (new_w - w) // 2
        cropped = resized[start_y:start_y+h, start_x:start_x+w]
        return cropped
    else:  # Zoom out
        new_h, new_w = int(h * zoom_factor), int(w * zoom_factor)
        resized = cv2.resize(image, (new_w, new_h))
        canvas = np.zeros((h, w, 3), dtype=np.uint8)
        start_y = (h - new_h) // 2
        start_x = (w - new_w) // 2
        canvas[start_y:start_y+new_h, start_x:start_x+new_w] = resized
        return canvas


def distance_metric(vec1, vec2, metric='euclidean'):
    """Calculate distance between two feature vectors"""
    if metric == 'euclidean':
        return np.linalg.norm(vec1 - vec2)
    elif metric == 'manhattan':
        return np.sum(np.abs(vec1 - vec2))
    elif metric == 'cosine':
        dot = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        return 1.0 - (dot / (norm1 * norm2 + 1e-9))
    elif metric == 'mse':
        return np.mean((vec1 - vec2) ** 2)
    else:
        return np.linalg.norm(vec1 - vec2)


def knn_classify(test_features, training_samples, k=3, metric='euclidean'):
    """KNN classifier using pairwise distance features"""
    distances = []
    for sample in training_samples:
        train_features = np.array(sample['features'])
        dist = distance_metric(test_features, train_features, metric)
        distances.append((dist, sample['class'], sample.get('score', 1.0)))
    
    distances.sort(key=lambda x: x[0])
    top_k = distances[:k]
    
    votes = [cls for _, cls, _ in top_k]
    vote_counts = Counter(votes)
    
    if len(vote_counts) == 1:
        return top_k[0][1], top_k[0][0]
    
    max_count = max(vote_counts.values())
    tied = [cls for cls, count in vote_counts.items() if count == max_count]
    
    if len(tied) == 1:
        winning_class = tied[0]
        winning_dist = min(dist for dist, cls, _ in top_k if cls == winning_class)
        return winning_class, winning_dist
    
    best_class = None
    best_score = -1
    for cls in tied:
        avg_score = np.mean([score for _, c, score in top_k if c == cls])
        if avg_score > best_score:
            best_score = avg_score
            best_class = cls
    
    winning_dist = min(dist for dist, cls, _ in top_k if cls == best_class)
    return best_class, winning_dist


def evaluate_with_zoom_voting(dataset_dir, training_data, k=3, metric='euclidean', min_detection_confidence=0.5):
    """Evaluate with zoom voting (original, +10%, -10%)"""
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(max_num_hands=1,
                           min_detection_confidence=min_detection_confidence,
                           min_tracking_confidence=0.5)

    training_samples = training_data['samples']
    
    results = []
    correct = 0
    total = 0
    no_hand = 0
    
    vote_agreed = 0
    vote_disagreed = 0
    
    # Per-class statistics
    class_stats = {}
    
    class_dirs = sorted([d for d in os.listdir(dataset_dir) 
                        if os.path.isdir(os.path.join(dataset_dir, d))])

    # Count total images for progress tracking
    total_images = 0
    for class_name in class_dirs:
        class_path = os.path.join(dataset_dir, class_name)
        image_files = [f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        total_images += len(image_files)
    
    processed_images = 0
    last_update_time = time.time()
    start_time = time.time()
    print(f"Total images to evaluate: {total_images}")
    print()

    for class_name in class_dirs:
        class_path = os.path.join(dataset_dir, class_name)
        image_files = sorted([f for f in os.listdir(class_path) 
                            if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

        # Initialize class statistics
        if class_name not in class_stats:
            class_stats[class_name] = {'total': 0, 'correct': 0, 'incorrect': 0, 'no_hand': 0}

        for img_file in image_files:
            processed_images += 1
            
            # Print progress every 10 seconds or on last image
            current_time = time.time()
            if (current_time - last_update_time >= 10) or processed_images == total_images:
                elapsed = current_time - start_time
                percentage = (processed_images / total_images) * 100
                images_per_sec = processed_images / elapsed if elapsed > 0 else 0
                eta_seconds = (total_images - processed_images) / images_per_sec if images_per_sec > 0 else 0
                print(f"Progress: {processed_images}/{total_images} ({percentage:.1f}%) - Accuracy: {correct}/{total} ({(correct/total*100 if total > 0 else 0):.1f}%) - Speed: {images_per_sec:.1f} img/s - ETA: {int(eta_seconds)}s")
                last_update_time = current_time
            
            img_path = os.path.join(class_path, img_file)
            image = cv2.imread(img_path)
            if image is None:
                continue

            total += 1
            
            # Test with 3 zoom levels
            zoom_factors = [1.0, 1.1, 0.9]  # original, +10%, -10%
            predictions = []
            confidences = []
            
            for zoom in zoom_factors:
                test_img = apply_zoom(image, zoom)
                rgb = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)
                test_results = hands.process(rgb)
                
                if not test_results.multi_hand_landmarks:
                    continue
                
                hand_lm = test_results.multi_hand_landmarks[0]
                test_features = get_pairwise_distances(hand_lm)
                pred_class, pred_dist = knn_classify(test_features, training_samples, k, metric)
                
                predictions.append(pred_class)
                confidences.append(1.0 / (pred_dist + 1e-6))
            
            if len(predictions) == 0:
                no_hand += 1
                class_stats[class_name]['total'] += 1
                class_stats[class_name]['no_hand'] += 1
                results.append({
                    'image': img_file,
                    'true_class': class_name,
                    'predicted_class': 'NO_HAND',
                    'correct': False
                })
                continue
            
            # Voting logic
            vote_counts = Counter(predictions)
            max_votes = max(vote_counts.values())
            
            if len(set(predictions)) == 1:
                # All agree
                final_pred = predictions[0]
                vote_agreed += 1
            elif max_votes >= 2:
                # 2/3 majority
                final_pred = [cls for cls, count in vote_counts.items() if count == max_votes][0]
                vote_agreed += 1
            else:
                # All different - use confidence
                best_idx = np.argmax(confidences)
                final_pred = predictions[best_idx]
                vote_disagreed += 1
            
            is_correct = (final_pred == class_name)
            if is_correct:
                correct += 1
                class_stats[class_name]['correct'] += 1
            else:
                class_stats[class_name]['incorrect'] += 1
            
            class_stats[class_name]['total'] += 1

            results.append({
                'image': img_file,
                'true_class': class_name,
                'predicted_class': final_pred,
                'correct': is_correct,
                'votes': predictions,
                'confidences': [float(c) for c in confidences]
            })

    hands.close()
    
    print(f"Voting stats: Agreed={vote_agreed}, Disagreed={vote_disagreed}, Used confidence={vote_disagreed}")
    
    return results, correct, total, no_hand, class_stats


def main():
    parser = argparse.ArgumentParser(description='Evaluate using pairwise distances (21x21)')
    parser.add_argument('dataset', help='Path to test dataset directory')
    parser.add_argument('training_json', help='Path to training JSON file')
    parser.add_argument('--k', type=int, default=3, help='K for KNN (default: 3)')
    parser.add_argument('--metric', default='euclidean', choices=['euclidean', 'manhattan', 'cosine', 'mse'])
    parser.add_argument('--zoom-voting', action='store_true', help='Use zoom voting (original, +10%, -10%)')
    parser.add_argument('--out-json', help='Output JSON results file')
    parser.add_argument('--out-csv', help='Output CSV predictions file')

    args = parser.parse_args()

    # Load training data
    with open(args.training_json, 'r') as f:
        training_data = json.load(f)

    print(f"Loaded {len(training_data['samples'])} training samples (pairwise distances 21x21)")
    
    if args.zoom_voting:
        print("Using zoom voting: original, +10% zoom, -10% zoom")
        results, correct, total, no_hand, class_stats = evaluate_with_zoom_voting(
            args.dataset, training_data, args.k, args.metric
        )
    else:
        print("Error: Non-zoom voting not implemented. Use --zoom-voting")
        return

    # Write results
    if args.out_json:
        output = {
            'results': results,
            'total': total,
            'correct': correct,
            'no_hand_detected': no_hand,
            'accuracy_all': correct / total if total > 0 else 0,
            'accuracy_detected': correct / (total - no_hand) if (total - no_hand) > 0 else 0,
            'k': args.k,
            'metric': args.metric,
            'feature_type': 'pairwise_distances_21x21',
            'class_statistics': class_stats
        }
        with open(args.out_json, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"Wrote results to {args.out_json}")

    if args.out_csv:
        with open(args.out_csv, 'w') as f:
            f.write("image,true_class,predicted_class,correct\n")
            for r in results:
                f.write(f"{r['image']},{r['true_class']},{r['predicted_class']},{r['correct']}\n")

    # Print summary
    print(f"\nKNN Classifier with Pairwise Distances 21x21 (K={args.k}, metric={args.metric})")
    print(f"Total: {total}")
    print(f"Correct: {correct}")
    print(f"No hand detected: {no_hand}")
    print(f"Accuracy (all): {correct/total if total > 0 else 0}")
    print(f"Accuracy (detected only): {correct/(total-no_hand) if (total-no_hand) > 0 else 0}")
    
    # Print per-class statistics
    print(f"\nPer-Class Statistics:")
    print(f"{'Class':<10} {'Total':<8} {'Correct':<10} {'Incorrect':<12} {'No Hand':<10} {'Accuracy':<10}")
    print("-" * 70)
    for cls in sorted(class_stats.keys()):
        stats = class_stats[cls]
        accuracy = (stats['correct'] / stats['total'] * 100) if stats['total'] > 0 else 0
        print(f"{cls:<10} {stats['total']:<8} {stats['correct']:<10} {stats['incorrect']:<12} {stats['no_hand']:<10} {accuracy:.1f}%")


if __name__ == '__main__':
    main()
