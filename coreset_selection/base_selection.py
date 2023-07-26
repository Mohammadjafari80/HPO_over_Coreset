import numpy as np
import os
import torch
from feature_extractor import get_pretrained_model
from utils import get_features

class CoresetSelection:

    def __init__(self, model_name, dataset, dataset_name, count, save=True, device='cuda'):
        self.model_name = model_name
        self.dataset = dataset
        self.dataset_name = dataset_name
        self.count = count
        self.save = save
        self.device = device

        os.makedirs('./features/', exist_ok=True)
        os.makedirs('./coreset_indices/', exist_ok=True)

        
        self.count = np.min(np.array([self.count, len(self.dataset)]))

        self.save_features_path = os.path.join(f'./features/{self.dataset_name}_{self.model_name}.npy')
        self.save_targets_path = os.path.join(f'./features/{self.dataset_name}_targets.npy')

    def get_features_targets(self):
        features, targets = None, None
        
        if os.path.exists(self.save_features_path):
            features = np.load(self.save_features_path)
            targets = np.load(self.save_targets_path)
            
        else:
            model = get_pretrained_model(self.model_name)
            train_loader = torch.utils.data.DataLoader(self.dataset, batch_size=16, shuffle=False, num_workers=2)
            features, targets = get_features(model, train_loader, self.device)
            
        features = features.squeeze()

        return features, targets

    def get_coreset(self):
        raise NotImplementedError