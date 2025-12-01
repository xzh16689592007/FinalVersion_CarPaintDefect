# predict_yolo.py - 用YOLO模型做缺陷分割（只检测位置，不分类）

import os
import argparse
import cv2
import numpy as np
from ultralytics import YOLO


# 缺陷框颜色配置
DEFECT_COLORS = {
    'defect': (0, 255, 0),      # 单类别：绿色
    'dirt': (0, 0, 255),        # 多类别
    'runs': (0, 165, 255),
    'scratch': (0, 255, 255),
    'water marks': (255, 0, 0),
}
DEFAULT_COLOR = (0, 255, 0)


def cv2_imread(img_path):
    return cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)


def cv2_imwrite(output_path, img):
    ext = os.path.splitext(output_path)[1]
    cv2.imencode(ext, img)[1].tofile(output_path)


def predict_yolo(weights, input_path, output_dir, conf=0.25, iou=0.45):
    """
    用YOLO模型做检测自动画框
    
    conf: 置信度阈值
    iou: NMS的IOU阈值
    """
    # 加载模型
    model = YOLO(weights)
    
    # 获取类别名
    names = model.names
    
    # 清空并创建输出目录
    if os.path.exists(output_dir):
        import shutil
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取图片列表
    if os.path.isdir(input_path):
        imgs = [os.path.join(input_path, f) for f in os.listdir(input_path) 
                if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    else:
        imgs = [input_path]

    print(f"\n{'='*60}")
    print(f"YOLO Detection - {len(imgs)} images")
    print(f"Model: {weights}")
    print(f"Confidence threshold: {conf}")
    print(f"Output: {output_dir}")
    print(f"{'='*60}\n")
    
    total_defects = 0
    
    for img_path in imgs:
        # 读取图片
        img = cv2_imread(img_path)
        if img is None:
            print(f"Warning: Cannot read {img_path}")
            continue
        
        h, w = img.shape[:2]
        
        # YOLO推理
        results = model.predict(img_path, conf=conf, iou=iou, verbose=False)
        result = results[0]
        
        # 创建带信息栏的图片
        info_height = 80
        result_img = np.zeros((h + info_height, w, 3), dtype=np.uint8)
        result_img[info_height:, :] = img
        result_img[:info_height, :] = (40, 40, 40)
        
        # 统计检测结果
        detected = {}
        boxes = result.boxes
        
        for box in boxes:
            cls_id = int(box.cls[0])
            cls_name = names[cls_id]
            conf_score = float(box.conf[0])
            
            # 统计
            if cls_name not in detected:
                detected[cls_name] = []
            detected[cls_name].append(conf_score)
            
            # 获取边界框坐标
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = int(x1), int(y1) + info_height, int(x2), int(y2) + info_height
            
            # 画框
            color = DEFECT_COLORS.get(cls_name, DEFAULT_COLOR)
            cv2.rectangle(result_img, (x1, y1), (x2, y2), color, 2)
            
            # 画标签背景（黑色背景 + 白色文字，更清晰）
            label = f"{cls_name} {conf_score:.2f}"
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(result_img, (x1, y1 - label_h - 12), (x1 + label_w + 8, y1), (0, 0, 0), -1)  # 黑色背景
            cv2.rectangle(result_img, (x1, y1 - label_h - 12), (x1 + label_w + 8, y1), color, 2)  # 彩色边框
            cv2.putText(result_img, label, (x1 + 4, y1 - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)  # 白色粗体文字
        
        # 顶部信息栏
        x_offset = 10
        cv2.putText(result_img, "YOLO Detected:", (x_offset, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        x_offset = 140
        
        if detected:
            for cls_name, scores in detected.items():
                color = DEFECT_COLORS.get(cls_name, DEFAULT_COLOR)
                text = f"{cls_name}:{len(scores)}"
                cv2.putText(result_img, text, (x_offset, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                x_offset += len(text) * 10 + 15
        else:
            cv2.putText(result_img, "No defects (OK)", (x_offset, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # 第二行
        cv2.putText(result_img, "YOLO model auto-detection (no label file needed)", 
                    (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        
        # 保存
        basename = os.path.basename(img_path)
        name_part, ext = os.path.splitext(basename)
        output_path = os.path.join(output_dir, f"{name_part}_yolo{ext}")
        cv2_imwrite(output_path, result_img)
        
        # 打印结果
        num_boxes = len(boxes)
        total_defects += num_boxes
        status = "DEFECT" if num_boxes > 0 else "OK"
        print(f"[{status}] {basename} - {num_boxes} defects")
        if detected:
            for cls_name, scores in detected.items():
                print(f"   {cls_name}: {len(scores)} ({', '.join([f'{s:.2f}' for s in scores])})")
        print(f"   Saved: {output_path}\n")
    
    print(f"{'='*60}")
    print(f"Done! Total {total_defects} defects in {len(imgs)} images")
    print(f"Results saved to: {output_dir}")
    print(f"{'='*60}")


def get_default_output_dir():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(script_dir, '..', 'yolo_results'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='YOLO Defect Detection')
    parser.add_argument('--weights', type=str, required=True, help='YOLO model weights (.pt)')
    parser.add_argument('--input', type=str, required=True, help='Input image or folder')
    parser.add_argument('--output', type=str, default=None, help='Output directory')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.45, help='NMS IOU threshold')
    args = parser.parse_args()
    
    output_dir = args.output if args.output else get_default_output_dir()
    
    predict_yolo(args.weights, args.input, output_dir, args.conf, args.iou)
