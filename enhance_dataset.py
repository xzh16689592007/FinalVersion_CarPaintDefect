# -*- coding: utf-8 -*-
"""
enhance_dataset.py - Dataset enhancement tool
Enhance all images in a dataset and save to new location
"""

import os
import argparse
import shutil
import cv2
import numpy as np
from tqdm import tqdm
from image_enhance import auto_enhance


def cv2_imread(img_path):
    """Read image with Chinese path support"""
    return cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)


def cv2_imwrite(output_path, img):
    """Write image with Chinese path support"""
    ext = os.path.splitext(output_path)[1]
    cv2.imencode(ext, img)[1].tofile(output_path)


def enhance_dataset(input_root, output_root):
    """
    Enhance all images in dataset
    
    Args:
        input_root: Input dataset root (with train/valid/test folders)
        output_root: Output dataset root
    """
    # Create output directory
    if os.path.exists(output_root):
        shutil.rmtree(output_root)
    os.makedirs(output_root, exist_ok=True)
    
    # Copy data.yaml
    yaml_src = os.path.join(input_root, 'data.yaml')
    if os.path.exists(yaml_src):
        shutil.copy(yaml_src, output_root)
    
    # Process each split
    splits = ['train', 'valid', 'test']
    
    for split in splits:
        images_dir = os.path.join(input_root, split, 'images')
        labels_dir = os.path.join(input_root, split, 'labels')
        
        if not os.path.exists(images_dir):
            continue
        
        # Create output directories
        out_images = os.path.join(output_root, split, 'images')
        out_labels = os.path.join(output_root, split, 'labels')
        os.makedirs(out_images, exist_ok=True)
        os.makedirs(out_labels, exist_ok=True)
        
        # Copy labels
        if os.path.exists(labels_dir):
            for f in os.listdir(labels_dir):
                shutil.copy(os.path.join(labels_dir, f), out_labels)
        
        # Enhance images
        image_files = [f for f in os.listdir(images_dir) 
                      if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        
        print(f"\n[{split}] Enhancing {len(image_files)} images...")
        
        for fname in tqdm(image_files, desc=split):
            img_path = os.path.join(images_dir, fname)
            img = cv2_imread(img_path)
            
            if img is None:
                continue
            
            # Apply auto enhancement
            enhanced, _, _ = auto_enhance(img)
            
            # Save
            out_path = os.path.join(out_images, fname)
            cv2_imwrite(out_path, enhanced)
    
    print(f"\nDone! Enhanced dataset: {output_root}")


def get_default_paths():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return {
        'input': os.path.abspath(os.path.join(script_dir, '..', '..')),
        'output': os.path.abspath(os.path.join(script_dir, '..', '..', '..', 
                                               'final year car paint defect.v1i.yolov11_enhanced'))
    }


if __name__ == '__main__':
    defaults = get_default_paths()
    
    parser = argparse.ArgumentParser(description='Enhance dataset images')
    parser.add_argument('--input', '-i', type=str, default=defaults['input'], 
                        help='Input dataset root')
    parser.add_argument('--output', '-o', type=str, default=defaults['output'], 
                        help='Output dataset root')
    args = parser.parse_args()
    
    enhance_dataset(args.input, args.output)
