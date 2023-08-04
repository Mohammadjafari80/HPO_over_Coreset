import torch
from torch.utils.data import Dataset
import os
import zipfile
import urllib.request
from torchvision.datasets import ImageFolder
from torchvision.datasets import CIFAR10, CIFAR100
import torchvision.transforms as transforms
import torch
from torch.utils.data import Dataset, Subset
import numpy as np
import requests
from PIL import Image
import pandas as pd
import pickle

class CustomSubset(Subset):
    def __init__(self, dataset, indices):
        super(CustomSubset, self).__init__(dataset, indices)
        self.classes = np.array(dataset.classes)
        self.targets = np.array([dataset.targets[i] for i in indices])
        self.data = np.array([dataset.data[i] for i in indices])
        
        
MEAN = {
    'cifar10': (0.4914, 0.4822, 0.4465),
    'cifar100': (0.5071, 0.4867, 0.4408),
    'tinyimagenet': (0.4802, 0.4481, 0.3975),
    'imagenet': (0.485, 0.456, 0.406)
}

STD = {
    'cifar10': (0.2023, 0.1994, 0.2010),
    'cifar100': (0.2675, 0.2565, 0.2761),
    'tinyimagenet': (0.2302, 0.2265, 0.2262),
    'imagenet': (0.229, 0.224, 0.225)
}

SIZE = {
    'cifar10': 32,
    'cifar100': 32,
    'tinyimagenet': 32,
    'imagenet': 224
}

class CustomWeightedDataset(Dataset):
    """Class for datasets with weight for each sample"""
    
    def __init__(self, dataset, weights):
        self.dataset = dataset
        self.weights = weights

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data, label = self.dataset[idx]
        weight = self.weights[idx]
        return data, label, weight


class TinyImageNet(Dataset):
    BASE_URL = 'http://cs231n.stanford.edu/tiny-imagenet-200.zip'
    FILE_NAME = 'tiny-imagenet-200.zip'

    def __init__(self, root, transform=None, train=True):
        self.root = root
        self.transform = transform
        self.train = train

        # Check if files are already downloaded
        if not os.path.exists(os.path.join(self.root, 'tiny-imagenet-200')):
            self.download()

        self.data, self.labels, self.label_to_idx = self.load_dataset()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx], self.labels[idx]

        # Load image
        img = Image.open(img_path)

        if self.transform:
            img = self.transform(img)

        return img, label

    def download(self):
        response = requests.get(self.BASE_URL, stream=True)
        file_size = int(response.headers['Content-Length'])
        chunk = 1
        chunk_size = 1024
        num_bars = int(file_size / chunk_size)

        with open(os.path.join(self.root, self.FILE_NAME), 'wb') as fp:
            for chunk in response.iter_content(chunk_size=chunk_size):
                fp.write(chunk)

        # Extract the zip file
        with zipfile.ZipFile(os.path.join(self.root, self.FILE_NAME), 'r') as zip_ref:
            zip_ref.extractall(self.root)

    def load_dataset(self):
        if self.train:
            data_dir = os.path.join(self.root, 'tiny-imagenet-200', 'train')
            folders = os.listdir(data_dir)
            data = []
            labels = []
            label_to_idx = {folder: i for i, folder in enumerate(folders)}

            for folder in folders:
                for file in os.listdir(os.path.join(data_dir, folder, 'images')):
                    file_path = os.path.join(data_dir, folder, 'images', file)
                    data.append(file_path)
                    labels.append(label_to_idx[folder])

            with open(os.path.join(self.root, 'label_to_idx.pkl'), 'wb') as f:
                pickle.dump(label_to_idx, f)
        else:
            data_dir = os.path.join(self.root, 'tiny-imagenet-200', 'val')
            annotations = pd.read_csv(os.path.join(data_dir, 'val_annotations.txt'), sep='\t', header=None, index_col=False)
            data = [os.path.join(data_dir, 'images', filename) for filename in annotations[0]]

            with open(os.path.join(self.root, 'label_to_idx.pkl'), 'rb') as f:
                label_to_idx = pickle.load(f)

            labels = [label_to_idx[label] for label in annotations[1]]

        return data, labels, label_to_idx
    

def get_classes_count(dataset_name):
    if dataset_name == 'cifar10':
        return 10
    elif dataset_name == 'cifar100':
        return 100
    elif dataset_name == 'tinyimagenet':
        return 200
    else:
        raise ValueError('Unknown dataset name: {}'.format(dataset_name))
    

DATASETS_DICT = {
    'cifar10': CIFAR10,
    'cifar100': CIFAR100,
    'TinyImageNet': TinyImageNet,
}

def get_transforms(dataset_name):

    mean, std = MEAN[dataset_name], STD[dataset_name]
    size = SIZE[dataset_name]
    
    normalize = transforms.Normalize(mean=mean, std=std)

    train_transform = transforms.Compose([
            transforms.Resize(size=size),
            transforms.Lambda(lambda img: img.convert('RGB')),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
    ])
    
    test_transform = transforms.Compose([
            transforms.Resize(size=size),
            transforms.Lambda(lambda img: img.convert('RGB')),
            transforms.ToTensor(),
            normalize,
    ])
    
    return train_transform, test_transform


def get_dataset(dataset_name, root, train=True, transform=None, download=True):
    if dataset_name == 'tinyimagenet':
        return TinyImageNet(root, train=train, transform=transform)
    elif dataset_name == 'cifar10':
        return CIFAR10(root, train=train, transform=transform, download=download)
    elif dataset_name == 'cifar100':
        return CIFAR100(root, train=train, transform=transform, download=download)
    else:
        raise ValueError('Unknown dataset name: {}'.format(dataset_name))