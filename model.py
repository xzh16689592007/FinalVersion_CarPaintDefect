# model.py - 模型定义
# 用ResNet18做多标签分类，改了最后一层fc

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights


def get_resnet18_multilabel(num_classes, pretrained=True):
#拿预训练的ResNet18改成多标签分类器,把最后的全连接层换成num_classes个输出
    weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.resnet18(weights=weights)
    in_feats = model.fc.in_features
    # 换掉最后一层，输出改成我们的类别数
    model.fc = nn.Linear(in_feats, num_classes)
    return model
