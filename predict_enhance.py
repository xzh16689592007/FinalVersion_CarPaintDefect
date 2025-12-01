# -*- coding: utf-8 -*-
"""
predict_enhance.py - Image enhancement prediction tool
Apply enhancement methods and visualize results
"""

import os
import argparse
import cv2
import numpy as np
from image_enhance import auto_enhance, enhance_image, evaluate_image


def cv2_imread(img_path):
    """Read image with Chinese path support"""
    return cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)


def cv2_imwrite(output_path, img):
    """Write image with Chinese path support"""
    ext = os.path.splitext(output_path)[1]
    cv2.imencode(ext, img)[1].tofile(output_path)


def predict_enhance(input_path, output_dir, method='auto'):
    """
    Apply enhancement to images
    
    Args:
        input_path: Input image or folder
        output_dir: Output directory
        method: Enhancement method (auto/clahe/hist/stretch/unsharp)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Get image list
    if os.path.isdir(input_path):
        imgs = [os.path.join(input_path, f) for f in os.listdir(input_path) 
                if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    else:
        imgs = [input_path]
    
    print(f"\n{'='*60}")
    print(f"Image Enhancement")
    print(f"{'='*60}")
    print(f"Images: {len(imgs)}")
    print(f"Method: {method}")
    print(f"Output: {output_dir}")
    print(f"{'='*60}\n")
    
    for img_path in imgs:
        img = cv2_imread(img_path)
        if img is None:
            print(f"[ERROR] Cannot read: {img_path}")
            continue
        
        basename = os.path.basename(img_path)
        
        # Apply enhancement
        if method == 'auto':
            enhanced, analysis, applied = auto_enhance(img)
            method_str = '+'.join([m[0] for m in applied])
        else:
            enhanced = enhance_image(img, method)
            method_str = method
        
        # Save result
        name_part, ext = os.path.splitext(basename)
        output_path = os.path.join(output_dir, f"{name_part}_enhanced{ext}")
        cv2_imwrite(output_path, enhanced)
        
        print(f"[OK] {basename} -> {method_str}")
    
    print(f"\n{'='*60}")
    print(f"Done! Results: {output_dir}")
    print(f"{'='*60}")


def get_default_output():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(script_dir, '..', 'enhance_results'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Image Enhancement')
    parser.add_argument('--input', '-i', type=str, required=True, help='Input image/folder')
    parser.add_argument('--output', '-o', type=str, default=get_default_output(), help='Output dir')
    parser.add_argument('--method', '-m', type=str, default='auto',
                        choices=['auto', 'clahe', 'hist', 'stretch', 'unsharp'],
                        help='Enhancement method')
    args = parser.parse_args()
    
    predict_enhance(args.input, args.output, args.method)
