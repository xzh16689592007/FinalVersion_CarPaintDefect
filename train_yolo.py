# train_yolo.py - 训练YOLOv11检测模型
# 能精确定位缺陷位置并画框

import os
import argparse
from ultralytics import YOLO


def train_yolo(data_yaml, epochs=50, imgsz=640, batch=16, model_size='n'):
    """
    训练YOLOv11检测模型
    
    model_size: n(nano), s(small), m(medium), l(large), x(xlarge)
    越大越准但越慢
    """
    # 加载预训练的YOLOv11模型
    model_name = f'yolo11{model_size}.pt'
    model = YOLO(model_name)
    
    # 训练
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project='runs/detect',
        name='car_defect',
        exist_ok=True,
        patience=10,  # 早停
        save=True,
        plots=True,
    )
    
    print(f"\n训练完成！")
    print(f"最佳模型保存在: runs/detect/car_defect/weights/best.pt")
    
    return results


def get_default_data_yaml():
    """找到data.yaml的路径"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(script_dir, '..', '..', 'data.yaml'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train YOLOv11 Detection Model')
    parser.add_argument('--data', type=str, default=None, help='Path to data.yaml')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size')
    parser.add_argument('--batch', type=int, default=16, help='Batch size')
    parser.add_argument('--model', type=str, default='n', choices=['n', 's', 'm', 'l', 'x'],
                        help='Model size: n(nano), s(small), m(medium), l(large), x(xlarge)')
    args = parser.parse_args()
    
    data_yaml = args.data if args.data else get_default_data_yaml()
    
    print(f"Data yaml: {data_yaml}")
    print(f"Model: YOLOv11{args.model}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch}")
    print(f"Image size: {args.imgsz}")
    
    train_yolo(data_yaml, args.epochs, args.imgsz, args.batch, args.model)
