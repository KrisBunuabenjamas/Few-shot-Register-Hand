import os
import argparse
from PIL import Image, ImageOps
import numpy as np


def apply_zoom_pil(img, zoom_factor):
    """Apply zoom to PIL image. zoom_factor > 1.0 = zoom in, < 1.0 = zoom out."""
    w, h = img.size
    
    if zoom_factor > 1.0:  # Zoom in - crop center
        new_w, new_h = int(w * zoom_factor), int(h * zoom_factor)
        resized = img.resize((new_w, new_h), Image.LANCZOS)
        left = (new_w - w) // 2
        top = (new_h - h) // 2
        return resized.crop((left, top, left + w, top + h))
    else:  # Zoom out - add padding
        new_w, new_h = int(w * zoom_factor), int(h * zoom_factor)
        resized = img.resize((new_w, new_h), Image.LANCZOS)
        # Create black background
        result = Image.new('RGB', (w, h), (0, 0, 0))
        left = (w - new_w) // 2
        top = (h - new_h) // 2
        result.paste(resized, (left, top))
        return result


def sample_and_zoom_only(src_dataset_dir, dest_augmented_dir, dest_remaining_dir, samples_per_class=10):
    """
    Split dataset into two folders:
    1. First N images with 9 zoom versions (augmented)
    2. Remaining images without augmentation
    """
    os.makedirs(dest_augmented_dir, exist_ok=True)
    os.makedirs(dest_remaining_dir, exist_ok=True)
    
    for cls in sorted(os.listdir(src_dataset_dir)):
        src_cls_dir = os.path.join(src_dataset_dir, cls)
        if not os.path.isdir(src_cls_dir):
            continue
        
        dest_aug_cls_dir = os.path.join(dest_augmented_dir, cls)
        dest_rem_cls_dir = os.path.join(dest_remaining_dir, cls)
        os.makedirs(dest_aug_cls_dir, exist_ok=True)
        os.makedirs(dest_rem_cls_dir, exist_ok=True)
        
        # Get all image files
        imgs = [f for f in sorted(os.listdir(src_cls_dir)) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        imgs_augmented = imgs[:samples_per_class]
        imgs_remaining = imgs[samples_per_class:]
        
        print(f"  {cls}: {len(imgs_augmented)} augmented (×9 zoom) = {len(imgs_augmented) * 9}, {len(imgs_remaining)} remaining (no augment)")
        
        # Part 1: Create 9 zoom versions for first N images
        for img_name in imgs_augmented:
            src_path = os.path.join(src_cls_dir, img_name)
            img = Image.open(src_path)
            
            # Get base name without extension
            base_name = os.path.splitext(img_name)[0]
            ext = os.path.splitext(img_name)[1]
            
            # 9 zoom levels: 140%, 130%, 120%, 110% (in), 100%, 90%, 80%, 70%, 60% (out)
            zoom_versions = [
                (img, 'zoom100'),
                (apply_zoom_pil(img, 1.4), 'zoom140'),
                (apply_zoom_pil(img, 1.3), 'zoom130'),
                (apply_zoom_pil(img, 1.2), 'zoom120'),
                (apply_zoom_pil(img, 1.1), 'zoom110'),
                (apply_zoom_pil(img, 0.9), 'zoom90'),
                (apply_zoom_pil(img, 0.8), 'zoom80'),
                (apply_zoom_pil(img, 0.7), 'zoom70'),
                (apply_zoom_pil(img, 0.6), 'zoom60')
            ]
            
            for zoomed_img, zoom_name in zoom_versions:
                dest_name = f"{base_name}_{zoom_name}{ext}"
                dest_path = os.path.join(dest_aug_cls_dir, dest_name)
                zoomed_img.save(dest_path)
            
            img.close()
        
        # Part 2: Copy remaining images without augmentation
        for img_name in imgs_remaining:
            src_path = os.path.join(src_cls_dir, img_name)
            dest_path = os.path.join(dest_rem_cls_dir, img_name)
            img = Image.open(src_path)
            img.save(dest_path)
            img.close()


def main():
    parser = argparse.ArgumentParser(description='Split dataset: augmented (9 zooms) + remaining (no augment)')
    parser.add_argument('--src', required=True, help='Source dataset directory')
    parser.add_argument('--dest-augmented', required=True, help='Destination for augmented images (with zoom)')
    parser.add_argument('--dest-remaining', required=True, help='Destination for remaining images (no augment)')
    parser.add_argument('--samples', type=int, default=10, help='Number of images to augment per gesture (default: 10)')
    args = parser.parse_args()
    
    print(f"Splitting dataset with {args.samples} augmented images per gesture...")
    print(f"  First {args.samples}: 9 zooms (140%, 130%, 120%, 110% in, 100%, 90%, 80%, 70%, 60% out)")
    print(f"  Remaining: No augmentation (original only)")
    print(f"Source: {args.src}")
    print(f"Destination (augmented): {args.dest_augmented}")
    print(f"Destination (remaining): {args.dest_remaining}")
    print()
    
    sample_and_zoom_only(args.src, args.dest_augmented, args.dest_remaining, args.samples)
    
    print()
    print(f"✅ Done!")
    print(f"   Augmented dataset: {args.dest_augmented}")
    print(f"   Remaining dataset: {args.dest_remaining}")


if __name__ == '__main__':
    main()
