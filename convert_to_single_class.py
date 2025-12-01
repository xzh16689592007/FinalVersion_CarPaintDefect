# convert_to_single_class.py
# 将多类别数据集转换为单类别（所有缺陷合并为一类）
# 用于训练只检测"有无缺陷"的模型

import os
import argparse
import shutil
import yaml


def convert_labels(input_dir, output_dir):
    """
    将所有标签文件中的类别ID都改成0
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for filename in os.listdir(input_dir):
        if not filename.endswith('.txt'):
            continue
        
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)
        
        with open(input_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        new_lines = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            # 将类别ID改为0（统一为defect类）
            parts[0] = '0'
            new_lines.append(' '.join(parts) + '\n')
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.writelines(new_lines)
    
    print(f"转换完成: {input_dir} -> {output_dir}")


def copy_images(input_dir, output_dir):
    """复制图像文件"""
    os.makedirs(output_dir, exist_ok=True)
    
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            src = os.path.join(input_dir, filename)
            dst = os.path.join(output_dir, filename)
            shutil.copy2(src, dst)
    
    print(f"复制图像: {input_dir} -> {output_dir}")


def create_data_yaml(output_root):
    """创建新的data.yaml配置文件"""
    yaml_content = {
        'path': output_root,
        'train': 'train/images',
        'val': 'valid/images',
        'test': 'test/images',
        'nc': 1,
        'names': ['defect']
    }
    
    yaml_path = os.path.join(output_root, 'data.yaml')
    with open(yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(yaml_content, f, default_flow_style=False, allow_unicode=True)
    
    print(f"创建配置: {yaml_path}")


def main(input_root, output_root):
    """
    主函数：转换整个数据集
    """
    print("="*60)
    print("转换多类别数据集为单类别（defect）")
    print("="*60)
    print(f"输入: {input_root}")
    print(f"输出: {output_root}")
    print("="*60)
    
    # 处理 train/valid/test 三个子集
    for split in ['train', 'valid', 'test']:
        input_images = os.path.join(input_root, split, 'images')
        input_labels = os.path.join(input_root, split, 'labels')
        output_images = os.path.join(output_root, split, 'images')
        output_labels = os.path.join(output_root, split, 'labels')
        
        if os.path.exists(input_images):
            copy_images(input_images, output_images)
        
        if os.path.exists(input_labels):
            convert_labels(input_labels, output_labels)
    
    # 创建新的data.yaml
    create_data_yaml(output_root)
    
    print("="*60)
    print("转换完成！")
    print(f"新数据集: {output_root}")
    print("类别: 1类 (defect)")
    print("="*60)
    print("\n使用方法:")
    print(f"  python train_yolo.py --data \"{os.path.join(output_root, 'data.yaml')}\" --epochs 30")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='转换为单类别数据集')
    parser.add_argument('--input', '-i', type=str, required=True,
                        help='原始数据集根目录')
    parser.add_argument('--output', '-o', type=str, required=True,
                        help='输出数据集根目录')
    args = parser.parse_args()
    
    main(args.input, args.output)
