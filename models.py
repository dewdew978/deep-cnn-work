import torch
import torch.nn as nn
from torchvision.models import (
    resnet50, ResNet50_Weights,
    resnet18, ResNet18_Weights
)

def get_model(model_name="resnet50", num_classes=72, pretrained=True, freeze_backbone=False):
    """
    Builds a ResNet CNN model with Transfer Learning for Thai Character Classification.
    
    Args:
        model_name: 'resnet50' (default) or 'resnet18'
        num_classes: number of output classes (72 classes)
        pretrained: whether to load ImageNet pre-trained weights
        freeze_backbone: whether to freeze feature extractor layers initially
    """
    model_name = model_name.lower()
    
    if model_name == "resnet50":
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        model = resnet50(weights=weights)
        if freeze_backbone:
            for param in model.parameters():
                param.requires_grad = False
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, num_classes)
        )
        
    elif model_name == "resnet18":
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        model = resnet18(weights=weights)
        if freeze_backbone:
            for param in model.parameters():
                param.requires_grad = False
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_features, num_classes)
        )
    else:
        raise ValueError(f"Unsupported model: {model_name}. Choose 'resnet50' or 'resnet18'.")
        
    return model
