"""
eval_iou.py - 计算检测结果与真实标注的IoU

评估YOLO检测框与ground truth标注框的交并比
"""
import os
import sys
import argparse
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import cv2


def imread_chinese(path):
    """读取含中文路径的图像"""
    with open(path, 'rb') as f:
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def load_yolo_labels(label_path, img_w, img_h):
    """
    加载YOLO格式标注文件
    返回: [(class_id, x1, y1, x2, y2), ...]
    """
    boxes = []
    if not os.path.exists(label_path):
        return boxes
    
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                cls_id = int(parts[0])
                # YOLO格式: class cx cy w h (归一化)
                cx, cy, w, h = map(float, parts[1:5])
                # 转换为像素坐标
                x1 = int((cx - w/2) * img_w)
                y1 = int((cy - h/2) * img_h)
                x2 = int((cx + w/2) * img_w)
                y2 = int((cy + h/2) * img_h)
                boxes.append((cls_id, x1, y1, x2, y2))
    return boxes


def compute_iou(box1, box2):
    """
    计算两个框的IoU
    box: (x1, y1, x2, y2)
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    # 交集面积
    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)
    inter_area = inter_w * inter_h
    
    # 并集面积
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = area1 + area2 - inter_area
    
    if union_area == 0:
        return 0.0
    
    return inter_area / union_area


def match_boxes(pred_boxes, gt_boxes, iou_threshold=0.5):
    """
    匹配预测框和真实框
    返回: [(pred_idx, gt_idx, iou, pred_cls, gt_cls), ...], unmatched_pred, unmatched_gt
    """
    matches = []
    used_gt = set()
    unmatched_pred = []
    
    for pred_idx, (pred_cls, px1, py1, px2, py2) in enumerate(pred_boxes):
        best_iou = 0
        best_gt_idx = -1
        
        for gt_idx, (gt_cls, gx1, gy1, gx2, gy2) in enumerate(gt_boxes):
            if gt_idx in used_gt:
                continue
            
            iou = compute_iou((px1, py1, px2, py2), (gx1, gy1, gx2, gy2))
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx
        
        if best_gt_idx >= 0 and best_iou >= iou_threshold:
            gt_cls = gt_boxes[best_gt_idx][0]
            matches.append((pred_idx, best_gt_idx, best_iou, pred_cls, gt_cls))
            used_gt.add(best_gt_idx)
        else:
            unmatched_pred.append((pred_idx, best_iou if best_gt_idx >= 0 else 0))
    
    unmatched_gt = [i for i in range(len(gt_boxes)) if i not in used_gt]
    
    return matches, unmatched_pred, unmatched_gt


def evaluate_iou(image_dir, label_dir, model_path, iou_threshold=0.5, conf_threshold=0.25):
    """
    评估所有图像的IoU
    """
    # 加载模型
    model = YOLO(model_path)
    class_names = model.names
    
    # 获取图像列表
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
        image_files.extend(Path(image_dir).glob(ext))
        image_files.extend(Path(image_dir).glob(ext.upper()))
    
    print("=" * 70)
    print(f"IoU 评估 (阈值: {iou_threshold})")
    print("=" * 70)
    print(f"图像目录: {image_dir}")
    print(f"标注目录: {label_dir}")
    print(f"模型: {model_path}")
    print(f"图像数量: {len(image_files)}")
    print("=" * 70)
    
    # 统计
    all_ious = []
    total_tp = 0  # True Positive (IoU >= threshold)
    total_fp = 0  # False Positive (预测框无匹配)
    total_fn = 0  # False Negative (漏检)
    class_ious = {i: [] for i in range(len(class_names))}
    
    results_detail = []
    
    for img_path in sorted(image_files):
        # 读取图像
        img = imread_chinese(str(img_path))
        if img is None:
            continue
        
        h, w = img.shape[:2]
        
        # 加载真实标注
        label_path = os.path.join(label_dir, img_path.stem + '.txt')
        gt_boxes = load_yolo_labels(label_path, w, h)
        
        # YOLO预测
        results = model.predict(img, conf=conf_threshold, verbose=False)
        pred_boxes = []
        if results and len(results) > 0:
            for box in results[0].boxes:
                cls_id = int(box.cls[0])
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                conf = float(box.conf[0])
                pred_boxes.append((cls_id, x1, y1, x2, y2, conf))
        
        # 匹配
        # 转换格式去掉conf
        pred_for_match = [(cls, x1, y1, x2, y2) for cls, x1, y1, x2, y2, conf in pred_boxes]
        matches, unmatched_pred, unmatched_gt = match_boxes(pred_for_match, gt_boxes, iou_threshold)
        
        # 统计
        tp = len(matches)
        fp = len(unmatched_pred)
        fn = len(unmatched_gt)
        
        total_tp += tp
        total_fp += fp
        total_fn += fn
        
        img_ious = []
        for pred_idx, gt_idx, iou, pred_cls, gt_cls in matches:
            all_ious.append(iou)
            img_ious.append(iou)
            class_ious[gt_cls].append(iou)
        
        # 记录详情
        img_result = {
            'name': img_path.name,
            'gt_count': len(gt_boxes),
            'pred_count': len(pred_boxes),
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'ious': img_ious,
            'matches': matches,
            'unmatched_pred': unmatched_pred
        }
        results_detail.append(img_result)
        
        # 打印单图结果
        avg_iou = np.mean(img_ious) if img_ious else 0
        status = "✓" if fn == 0 and fp == 0 else "!"
        
        if len(gt_boxes) > 0 or len(pred_boxes) > 0:
            print(f"{status} {img_path.name}")
            print(f"   GT: {len(gt_boxes)}, Pred: {len(pred_boxes)}, TP: {tp}, FP: {fp}, FN: {fn}")
            if img_ious:
                print(f"   IoU: {', '.join([f'{iou:.3f}' for iou in img_ious])} | Avg: {avg_iou:.3f}")
            
            # 显示未匹配的预测框（可能是假阳性或IoU不足）
            for pred_idx, best_iou in unmatched_pred:
                pred_cls = pred_for_match[pred_idx][0]
                print(f"   [FP] Pred {class_names[pred_cls]}: best IoU = {best_iou:.3f} < {iou_threshold}")
            
            # 显示漏检
            for gt_idx in unmatched_gt:
                gt_cls = gt_boxes[gt_idx][0]
                print(f"   [FN] Miss {class_names[gt_cls]}")
    
    # 总结
    print("\n" + "=" * 70)
    print("总体统计")
    print("=" * 70)
    
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"True Positive (IoU >= {iou_threshold}): {total_tp}")
    print(f"False Positive (误检): {total_fp}")
    print(f"False Negative (漏检): {total_fn}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    if all_ious:
        print(f"\n平均 IoU: {np.mean(all_ious):.4f}")
        print(f"最小 IoU: {np.min(all_ious):.4f}")
        print(f"最大 IoU: {np.max(all_ious):.4f}")
        print(f"IoU >= 0.5 比例: {sum(1 for iou in all_ious if iou >= 0.5) / len(all_ious) * 100:.1f}%")
        print(f"IoU >= 0.75 比例: {sum(1 for iou in all_ious if iou >= 0.75) / len(all_ious) * 100:.1f}%")
    
    # 按类别统计
    print("\n" + "-" * 40)
    print("按类别 IoU 统计:")
    print("-" * 40)
    for cls_id, ious in class_ious.items():
        if ious:
            print(f"  {class_names[cls_id]:15s}: Avg IoU = {np.mean(ious):.4f} ({len(ious)} boxes)")
    
    print("=" * 70)
    
    return {
        'tp': total_tp,
        'fp': total_fp,
        'fn': total_fn,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'mean_iou': np.mean(all_ious) if all_ious else 0,
        'all_ious': all_ious,
        'details': results_detail
    }


def main():
    parser = argparse.ArgumentParser(description='计算检测IoU')
    parser.add_argument('--images', type=str, required=True, help='图像目录')
    parser.add_argument('--labels', type=str, default=None, help='标注目录（默认与images同级的labels）')
    parser.add_argument('--model', type=str, default=None, help='YOLO模型路径')
    parser.add_argument('--iou', type=float, default=0.5, help='IoU阈值')
    parser.add_argument('--conf', type=float, default=0.25, help='置信度阈值')
    
    args = parser.parse_args()
    
    # 默认标注目录
    if args.labels is None:
        args.labels = os.path.join(os.path.dirname(args.images.rstrip('/\\')), 'labels')
    
    # 默认模型
    if args.model is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        args.model = os.path.join(script_dir, 'runs/detect/car_defect/weights/best.pt')
    
    evaluate_iou(args.images, args.labels, args.model, args.iou, args.conf)


if __name__ == '__main__':
    main()
