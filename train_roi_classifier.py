# -*- coding: utf-8 -*-
"""
train_roi_classifier.py - Train ROI classifier
Crop ROI from YOLO annotations and train single-label classifier
"""

import os
import argparse
import glob
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torch.optim import Adam
from torchvision import transforms
from tqdm import tqdm
from collections import Counter

from model import get_resnet18_multilabel


def cv2_imread(img_path):
    """Read image with Chinese path support"""
    return cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)


def load_data_yaml(root):
    """Load data.yaml config file"""
    import yaml
    yaml_path = os.path.join(root, "data.yaml")
    if not os.path.exists(yaml_path):
        yaml_path = os.path.join(os.path.dirname(root), "data.yaml")
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"data.yaml not found under {root}")
    with open(yaml_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class ROIDataset(Dataset):
    """Dataset that crops ROI from YOLO annotations"""
    
    def __init__(self, images_dir, labels_dir, class_names, transform=None, 
                 min_size=20, padding_ratio=0.1):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.class_names = class_names
        self.num_classes = len(class_names)
        self.min_size = min_size
        self.padding_ratio = padding_ratio
        
        self.transform = transform or transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        self.samples = []
        self._build_samples()
        
    def _build_samples(self):
        """Build sample list from annotation files"""
        label_files = glob.glob(os.path.join(self.labels_dir, "*.txt"))
        
        for lf in label_files:
            stem = os.path.splitext(os.path.basename(lf))[0]
            
            img_path = None
            for ext in (".jpg", ".jpeg", ".png", ".bmp"):
                p = os.path.join(self.images_dir, stem + ext)
                if os.path.exists(p):
                    img_path = p
                    break
            
            if img_path is None:
                continue
            
            with open(lf, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    parts = line.split()
                    if len(parts) < 5:
                        continue
                    
                    try:
                        class_id = int(float(parts[0]))
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])
                        
                        if 0 <= class_id < self.num_classes:
                            self.samples.append((img_path, class_id, (x_center, y_center, width, height)))
                    except (ValueError, IndexError):
                        continue
        
        print(f"  Loaded {len(self.samples)} ROI samples")
        
        class_counts = Counter(s[1] for s in self.samples)
        for i, name in enumerate(self.class_names):
            print(f"    {name}: {class_counts.get(i, 0)}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, class_id, bbox = self.samples[idx]
        
        img = cv2_imread(img_path)
        if img is None:
            img = np.zeros((224, 224, 3), dtype=np.uint8)
            roi = img
        else:
            h, w = img.shape[:2]
            x_center, y_center, bw, bh = bbox
            
            cx = int(x_center * w)
            cy = int(y_center * h)
            box_w = int(bw * w)
            box_h = int(bh * h)
            
            pad_w = int(box_w * self.padding_ratio)
            pad_h = int(box_h * self.padding_ratio)
            
            x1 = max(0, cx - box_w // 2 - pad_w)
            y1 = max(0, cy - box_h // 2 - pad_h)
            x2 = min(w, cx + box_w // 2 + pad_w)
            y2 = min(h, cy + box_h // 2 + pad_h)
            
            if x2 - x1 < self.min_size:
                x1 = max(0, cx - self.min_size // 2)
                x2 = min(w, cx + self.min_size // 2)
            if y2 - y1 < self.min_size:
                y1 = max(0, cy - self.min_size // 2)
                y2 = min(h, cy + self.min_size // 2)
            
            roi = img[y1:y2, x1:x2]
            
            if roi.size == 0:
                roi = img
        
        roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            roi_tensor = self.transform(roi_rgb)
        else:
            roi_tensor = torch.from_numpy(roi_rgb).permute(2, 0, 1).float() / 255.0
        
        return roi_tensor, class_id


def train_roi_classifier(args):
    """Train ROI classifier"""
    data_root = args.data_root
    
    data_yaml = load_data_yaml(data_root)
    class_names = data_yaml.get("names", ['dirt', 'runs', 'scratch', 'water marks'])
    num_classes = len(class_names)
    
    print(f"\n{'='*60}")
    print(f"Training ROI Classifier")
    print(f"{'='*60}")
    print(f"Data root: {data_root}")
    print(f"Classes: {class_names}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"{'='*60}\n")
    
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    print("Loading training data...")
    train_ds = ROIDataset(
        os.path.join(data_root, "train", "images"),
        os.path.join(data_root, "train", "labels"),
        class_names,
        transform=train_transform
    )
    
    print("\nLoading validation data...")
    val_ds = ROIDataset(
        os.path.join(data_root, "valid", "images"),
        os.path.join(data_root, "valid", "labels"),
        class_names,
        transform=val_transform
    )
    
    if len(train_ds) == 0:
        print("Error: No training samples found!")
        return
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    
    model = get_resnet18_multilabel(num_classes, pretrained=True)
    model.to(device)
    
    optimizer = Adam(model.parameters(), lr=args.lr)
    
    class_counts = Counter(s[1] for s in train_ds.samples)
    total = sum(class_counts.values())
    weights = torch.tensor([total / (class_counts.get(i, 1) * num_classes) 
                           for i in range(num_classes)], dtype=torch.float32).to(device)
    print(f"\nClass weights: {weights.cpu().numpy()}")
    
    criterion = nn.CrossEntropyLoss(weight=weights)
    
    save_dir = os.path.abspath(os.path.join(data_root, '..', '..', 'models'))
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "best_roi.pth")
    
    best_acc = 0.0
    
    print(f"\n{'='*60}")
    print("Training...")
    print(f"{'='*60}\n")
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total_samples = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for imgs, labels in pbar:
            imgs = imgs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * imgs.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)
            
            pbar.set_postfix(loss=loss.item(), acc=correct/total_samples)
        
        train_loss = running_loss / len(train_ds)
        train_acc = correct / total_samples
        
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        class_correct = [0] * num_classes
        class_total = [0] * num_classes
        
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(device)
                labels = labels.to(device)
                
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * imgs.size(0)
                
                _, predicted = torch.max(outputs, 1)
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)
                
                for i in range(labels.size(0)):
                    label = labels[i].item()
                    class_total[label] += 1
                    if predicted[i] == label:
                        class_correct[label] += 1
        
        val_loss = val_loss / len(val_ds) if len(val_ds) > 0 else 0
        val_acc = val_correct / val_total if val_total > 0 else 0
        
        print(f"Epoch {epoch}: train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
              f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")
        
        for i, name in enumerate(class_names):
            if class_total[i] > 0:
                acc = class_correct[i] / class_total[i]
                print(f"  {name}: {class_correct[i]}/{class_total[i]} = {acc:.4f}")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f"  Saved best model (acc={best_acc:.4f}) to {save_path}")
    
    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"Best validation accuracy: {best_acc:.4f}")
    print(f"Model saved to: {save_path}")
    print(f"{'='*60}")


def get_default_data_root():
    """Default to enhanced dataset"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    enhanced_root = os.path.abspath(os.path.join(script_dir, '..', '..', '..', 
                                                  'final year car paint defect.v1i.yolov11_enhanced'))
    if os.path.exists(enhanced_root):
        return enhanced_root
    return os.path.abspath(os.path.join(script_dir, '..', '..'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train ROI classifier')
    parser.add_argument('--data-root', type=str, default=None, 
                        help='Dataset root (default: enhanced dataset)')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    args = parser.parse_args()
    
    if args.data_root is None:
        args.data_root = get_default_data_root()
    
    train_roi_classifier(args)
