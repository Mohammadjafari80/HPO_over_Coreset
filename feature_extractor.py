import torch
from torchvision import models

feature_extractors = [
    'resnet18',
    'resnet34',
    'resnet50',
    'resnet101',
    'resnet152',
    'wide_resnet101_2',
    'wide_resnet50_2',
    'vit_b_16',
]

def get_pretrained_model(backbone='resnet18'):

    if backbone in feature_extractors:
        model = eval(f'models.{backbone}(pretrained=True)')
    else:
        raise ValueError(f'backbone {backbone} not supported')
    
    for param in model.parameters():
        param.requires_grad = False

    if 'vit' in backbone:
        model.heads = torch.nn.Identity()
    else:
        model = torch.nn.Sequential(*list(model.children())[:-1])
    
    return model