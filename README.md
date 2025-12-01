# 汽车漆面缺陷检测系统

基于深度学习的汽车漆面缺陷检测与分类系统，包含图像增强、缺陷检测、缺陷分类三个模块。

## 项目结构

```
finalyear_classification/
    src/
        image_enhance.py          图像增强库
        predict_enhance.py        图像增强预测
        enhance_dataset.py        数据集批量增强
        model.py                  ResNet18分类模型
        train_roi_classifier.py   ROI分类器训练
        train_yolo.py             YOLO检测器训练
        predict_yolo.py           YOLO检测预测
        eval_iou.py               IoU评估
        convert_to_single_class.py  数据集转单类别
        pipeline.py               完整流水线
        yolo11n.pt                YOLOv11n预训练权重
        runs/detect/car_defect/weights/best.pt  训练好的YOLO权重
    pipeline_results/             流水线输出
    yolo_results/                 YOLO检测输出
    requirements.txt
```

## 模型说明

YOLO检测器: src/runs/detect/car_defect/weights/best.pt，单类别缺陷定位
ROI分类器: models/best_roi.pth，缺陷类型识别

## 缺陷类别

dirt: 灰尘/脏污
runs: 流挂
scratch: 划痕
water marks: 水渍


## 一、图像增强模块

对图像进行预处理，提升缺陷的可见度。

增强方法:
clahe: CLAHE自适应直方图均衡，适合光照不均
hist: 全局直方图均衡化
stretch: 对比度拉伸
unsharp: USM锐化
auto: 自动分析并选择合适方法

使用方法

```powershell
cd src

python predict_enhance.py -i "image.jpg" -o "output" -m clahe

python predict_enhance.py -i "image_folder" -o "output" -m auto
```

数据集增强

```powershell
python enhance_dataset.py --input "原始数据集路径" --output "增强数据集路径"
```


## 二、缺陷检测模块

使用YOLOv11进行缺陷区域检测。采用单类别检测，只负责定位缺陷位置，不区分类型。

训练

```powershell
cd src

python train_yolo.py --data "数据集/data.yaml" --epochs 30 --name car_defect
```

预测

```powershell
python predict_yolo.py --weights "runs/detect/car_defect/weights/best.pt" --input "测试图像路径" --output "输出目录"
```

评估

```powershell
python eval_iou.py --images "测试图像目录" --labels "标签目录" --model "runs/detect/car_defect/weights/best.pt"
```


## 三、缺陷分类模块

对检测到的每个缺陷区域进行分类，识别具体缺陷类型。

训练方式: 采用ROI裁剪训练，从标注中提取边界框，裁剪出缺陷区域，在裁剪后的小图上训练分类器。这样训练和推理时输入一致，都是缺陷的局部图像。

训练

```powershell
cd src

python train_roi_classifier.py --data-root "增强数据集路径" --epochs 30
```

训练结果: 验证集准确率96.13%，测试集准确率93.88%


## 四、完整流水线

整合检测和分类模块，对输入图像进行端到端处理。

流程: 输入图像 > 可选增强 > YOLO检测 > 裁剪ROI > 分类 > 输出结果

使用方法

```powershell
cd src

python pipeline.py -i "增强数据集/test/images" -o "输出目录"

python pipeline.py -i "原始图像目录" -o "输出目录" -e auto

python pipeline.py -i "图像" -o "输出" --yolo "yolo权重" --cls "分类器权重"
```

输出: 标注后的图像（检测框按缺陷类型用不同颜色绘制）和pipeline_report.txt检测分类报告


## 快速开始

```powershell
pip install -r requirements.txt

cd src
python pipeline.py -i "测试图像目录" -o "../pipeline_results"
```

## 从零训练

```powershell
cd src

python enhance_dataset.py --input "原始数据集" --output "增强数据集"

python convert_to_single_class.py -i "增强数据集" -o "单类别数据集"

python train_yolo.py --data "单类别数据集/data.yaml" --epochs 30

python train_roi_classifier.py --data-root "增强数据集" --epochs 30

python pipeline.py -i "增强数据集/test/images" -o "../pipeline_results"
```

## 依赖

Python 3.8+, PyTorch, ultralytics, OpenCV, torchvision, scikit-image
