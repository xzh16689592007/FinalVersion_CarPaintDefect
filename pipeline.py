# pipeline.py - 完整流水线：增强 → 分割 → 分类

import os
import argparse
import cv2
import numpy as np
import torch
from ultralytics import YOLO

from image_enhance import auto_enhance, enhance_image, evaluate_image
from model import get_resnet18_multilabel


CLASS_NAMES = ['dirt', 'runs', 'scratch', 'water marks']


DEFECT_COLORS = {
    'defect': (0, 255, 0),      # 单类别模型
    'dirt': (0, 0, 255),        # 多类别模型
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


def extract_contour_in_box(img, x1, y1, x2, y2):
    """在检测框内提取缺陷轮廓"""
    roi = img[y1:y2, x1:x2]
    if roi.size == 0:
        return None
    
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    largest = max(contours, key=cv2.contourArea)
    largest = largest + np.array([x1, y1])
    return largest


def load_classifier(weights_path, device):
    """加载分类模型"""
    model = get_resnet18_multilabel(num_classes=len(CLASS_NAMES), pretrained=False)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def classify_image(model, img, device):
    """对图像进行分类"""
    from torchvision import transforms
    
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    tensor = transform(img_rgb).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(tensor)
        probs = torch.sigmoid(output).cpu().numpy()[0]
    
    return {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))}


def classify_roi(model, img, x1, y1, x2, y2, device):
    """对检测框内的区域进行分类，返回最可能的类别"""
    from torchvision import transforms
    
    # 裁剪ROI区域
    roi = img[y1:y2, x1:x2]
    if roi.size == 0:
        return 'defect', 0.0  # 默认返回
    
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    tensor = transform(roi_rgb).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(tensor)
        probs = torch.sigmoid(output).cpu().numpy()[0]
    
    # 找出概率最高的类别
    max_idx = int(np.argmax(probs))
    max_prob = float(probs[max_idx])
    max_class = CLASS_NAMES[max_idx]
    
    return max_class, max_prob


def run_pipeline(yolo_weights, cls_weights, input_path, output_dir, 
                 enhance_method='auto', conf=0.25):
    """
    完整流水线:
    1. 图像增强
    2. YOLO检测 + 轮廓提取
    3. 分类确认
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载模型
    yolo_model = YOLO(yolo_weights)
    yolo_names = yolo_model.names
    cls_model = load_classifier(cls_weights, device)
    
    # 准备输出目录
    if os.path.exists(output_dir):
        import shutil
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取图片
    if os.path.isdir(input_path):
        imgs = [os.path.join(input_path, f) for f in os.listdir(input_path) 
                if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    else:
        imgs = [input_path]

    print(f"\n{'='*60}")
    print(f"Pipeline: Enhance -> Segment -> Classify")
    print(f"{'='*60}")
    print(f"Images: {len(imgs)}")
    print(f"Enhance: {enhance_method}")
    print(f"YOLO: {yolo_weights}")
    print(f"Classifier: {cls_weights}")
    print(f"Output: {output_dir}")
    print(f"{'='*60}\n")
    
    total_defects = 0
    report = []
    
    for img_path in imgs:
        img_orig = cv2_imread(img_path)
        if img_orig is None:
            continue
        
        basename = os.path.basename(img_path)
        h, w = img_orig.shape[:2]
        
        # === 1. 图像增强 ===
        if enhance_method == 'auto':
            img_enhanced, analysis, applied = auto_enhance(img_orig)
            enhance_info = '+'.join([m[0] for m in applied])
        elif enhance_method == 'none':
            img_enhanced = img_orig.copy()
            enhance_info = 'none'
        else:
            img_enhanced = enhance_image(img_orig, enhance_method)
            enhance_info = enhance_method
        
        # === 2. YOLO检测(用增强后的图) - 只负责分割，找出缺陷位置 ===
        temp_path = os.path.join(output_dir, '_temp.jpg')
        cv2_imwrite(temp_path, img_enhanced)
        results = yolo_model.predict(temp_path, conf=conf, verbose=False)
        os.remove(temp_path)
        result = results[0]
        
        # === 绘制结果 ===
        info_height = 100
        result_img = np.zeros((h + info_height, w, 3), dtype=np.uint8)
        result_img[info_height:, :] = img_enhanced.copy()
        result_img[:info_height, :] = (40, 40, 40)
        
        # === 3. 对每个检测框单独分类 ===
        detected = {}  # {类别名: [(置信度, 分类置信度), ...]}
        boxes = result.boxes
        
        for box in boxes:
            # YOLO只给出位置（单类别模型输出defect）
            det_conf = float(box.conf[0])
            bx1, by1, bx2, by2 = box.xyxy[0].cpu().numpy().astype(int)
            
            # 对这个框内的区域进行分类，确定具体缺陷类型
            cls_name, cls_prob = classify_roi(cls_model, img_enhanced, bx1, by1, bx2, by2, device)
            
            # 统计
            if cls_name not in detected:
                detected[cls_name] = []
            detected[cls_name].append((det_conf, cls_prob))
            
            # 根据分类结果选择颜色
            color = DEFECT_COLORS.get(cls_name, DEFAULT_COLOR)
            
            # 画矩形框
            cv2.rectangle(result_img, 
                          (bx1, by1 + info_height), 
                          (bx2, by2 + info_height), color, 2)
            
            # 标签：显示分类结果和置信度
            label = f"{cls_name} {cls_prob:.2f}"
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(result_img, (bx1, by1 + info_height - lh - 8), (bx1 + lw + 4, by1 + info_height), (0,0,0), -1)
            cv2.putText(result_img, label, (bx1 + 2, by1 + info_height - 4), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
        
        # 信息栏第一行：分割结果（检测到多少个框）
        x_offset = 10
        cv2.putText(result_img, "Segment:", (x_offset, 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        x_offset = 90
        if detected:
            total_boxes = sum(len(v) for v in detected.values())
            cv2.putText(result_img, f"{total_boxes} defects found", (x_offset, 25), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
        else:
            cv2.putText(result_img, "None", (x_offset, 25), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
        
        # 信息栏第二行：分类结果（每个框的分类统计）
        x_offset = 10
        cv2.putText(result_img, "Classify:", (x_offset, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        x_offset = 90
        for cls_name, scores_list in detected.items():
            color = DEFECT_COLORS.get(cls_name, DEFAULT_COLOR)
            text = f"{cls_name}:{len(scores_list)}"
            cv2.putText(result_img, text, (x_offset, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            x_offset += len(text) * 9 + 10
        
        # 信息栏第三行：增强方法
        cv2.putText(result_img, f"Enhanced: {enhance_info}", (10, 75), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)
        
        # 保存
        name_part, ext = os.path.splitext(basename)
        output_path = os.path.join(output_dir, f"{name_part}_pipeline{ext}")
        cv2_imwrite(output_path, result_img)
        
        num_defects = len(boxes)
        total_defects += num_defects
        
        # 记录
        # 统计每类的平均分类置信度
        cls_summary = {}
        for cls_name, scores_list in detected.items():
            avg_prob = sum(s[1] for s in scores_list) / len(scores_list)
            cls_summary[cls_name] = round(avg_prob, 3)
        
        entry = {
            'file': basename,
            'enhance': enhance_info,
            'yolo_detections': {k: len(v) for k, v in detected.items()},
            'classification': cls_summary
        }
        report.append(entry)
        
        status = "DEFECT" if num_defects > 0 else "OK"
        print(f"[{status}] {basename}")
        print(f"   Enhanced: {enhance_info}")
        print(f"   YOLO: {num_defects} defects")
        cls_str = ', '.join([f"{k}:{len(v)}" for k, v in detected.items()])
        print(f"   Classify: {cls_str if cls_str else 'none'}\n")
    
    # 保存报告
    save_report(output_dir, report, total_defects, len(imgs))
    
    print(f"{'='*60}")
    print(f"Done! {total_defects} defects in {len(imgs)} images")
    print(f"Results: {output_dir}")
    print(f"{'='*60}")


def save_report(output_dir, report, total_defects, total_imgs):
    """保存报告"""
    rpt_path = os.path.join(output_dir, 'pipeline_report.txt')
    with open(rpt_path, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write("       Pipeline Report: Enhance -> Segment -> Classify\n")
        f.write("="*60 + "\n\n")
        f.write(f"Total images: {total_imgs}\n")
        f.write(f"Total defects: {total_defects}\n\n")
        
        for r in report:
            f.write(f"[{r['file']}]\n")
            f.write(f"  Enhance: {r['enhance']}\n")
            f.write(f"  YOLO: {r['yolo_detections']}\n")
            f.write(f"  Class: {r['classification']}\n\n")
    
    print(f"Report: {rpt_path}")


def get_default_paths():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return {
        'yolo': os.path.join(script_dir, 'runs/detect/car_defect/weights/best.pt'),
        'cls': r'D:\图像处理基础开发\models\best_roi.pth',  # 使用ROI分类器
        'output': os.path.abspath(os.path.join(script_dir, '..', 'pipeline_results'))
    }


if __name__ == '__main__':
    defaults = get_default_paths()
    
    parser = argparse.ArgumentParser(description='Pipeline: Enhance -> Segment -> Classify')
    parser.add_argument('--input', '-i', type=str, required=True, help='Input image/folder')
    parser.add_argument('--output', '-o', type=str, default=defaults['output'], help='Output dir')
    parser.add_argument('--yolo', type=str, default=defaults['yolo'], help='YOLO weights')
    parser.add_argument('--cls', type=str, default=defaults['cls'], help='Classifier weights')
    parser.add_argument('--enhance', '-e', type=str, default='none',
                        choices=['auto', 'none', 'clahe', 'hist', 'stretch', 'unsharp'],
                        help='Enhance method (default: none, use auto for raw images)')
    parser.add_argument('--conf', type=float, default=0.25, help='YOLO confidence')
    args = parser.parse_args()
    
    run_pipeline(args.yolo, args.cls, args.input, args.output, args.enhance, args.conf)
