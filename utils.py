import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
import wandb

from preact_resnet import PreActResNet18, PreActResNet34, PreActResNet50, PreActResNet101, PreActResNet152
from resnet import resnet20, resnet32, resnet44, resnet56, resnet110, resnet1202
from feature_extractor import get_pretrained_model

def set_cudnn(device='cuda'):
    torch.backends.cudnn.enabled = (device == 'cuda')
    torch.backends.cudnn.benchmark = (device == 'cuda')


def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def log_on_wandb(results):
    try:
        wandb.log(results)
    except:
        print("Failed to Log Results on WANDB!")
        
class WeightedCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(WeightedCrossEntropyLoss, self).__init__()

    def forward(self, outputs, labels, weights):
        loss = F.cross_entropy(outputs, labels, reduction='none')
        weighted_loss = torch.mean(loss * weights)
        return weighted_loss
    
def get_model(arch, num_classes):
    
    if arch.lower() == 'preactresnet18':
        return PreActResNet18(num_classes=num_classes)
    elif arch.lower() == 'preactresnet34':
        return PreActResNet34(num_classes=num_classes)
    elif arch.lower() == 'preactresnet50':
        return PreActResNet50(num_classes=num_classes)
    elif arch.lower() == 'preactresnet101':
        return PreActResNet101(num_classes=num_classes)
    elif arch.lower() == 'preactresnet152':
        return PreActResNet152(num_classes=num_classes)
    elif arch.lower() == 'resnet20':
        return resnet20(num_classes=num_classes)
    elif arch.lower() == 'resnet32':
        return resnet32(num_classes=num_classes)
    elif arch.lower() == 'resnet44':
        return resnet44(num_classes=num_classes)
    elif arch.lower() == 'resnet56':
        return resnet56(num_classes=num_classes)
    elif arch.lower() == 'resnet110':
        return resnet110(num_classes=num_classes)
    elif arch.lower() == 'resnet1202':
        return resnet1202(num_classes=num_classes)
    else:
        raise ValueError('Unknown model architecture: {}'.format(arch))
    

def get_features(model, trainloader, device):
    # obtain features of each sample
    model = model.to(device)

    targets, features = [], []
    for img, target in tqdm(trainloader, desc='Extracting features'):
        targets.extend(target.numpy().tolist())
        img = img.to(device)
        feature = model(img).detach().cpu().numpy()
        features.extend([feature[i] for i in range(feature.shape[0])])

    features = np.array(features)
    targets = np.array(targets)

    return features.squeeze(), targets

def broadcast_weights(model_name, coreset_weights, train_dataset, coreset, batch_size=128, num_workers=4, device='cuda'):
    model = get_pretrained_model(model_name)

    coreset_loader = torch.utils.data.DataLoader(coreset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    full_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    # Get the coreset features and full features
    coreset_features, _ = get_features(model, coreset_loader, device)
    full_features, _ = get_features(model, full_loader, device)

    # Compute the nearest neighbors of each full feature in the coreset features
    cls = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(coreset_features)  # k=1 to get the closest neighbor
    _, indices = cls.kneighbors(full_features)

    # Initialize an array to store the full_weights for each full sample
    full_weights = np.zeros(len(full_features))

    # Assign weights of coreset samples to corresponding full samples
    full_weights = coreset_weights[indices[:, 0]]

    return full_weights



def get_histogram(weights):
    # Compute the histogram using np.histogram
    hist = np.histogram(weights, bins=20, range=(0, 1))

    return wandb.Histogram(np_histogram=hist)