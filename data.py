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
    'tinyimagenet': 64,
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


class TinyImageNetDataset(ImageFolder):
    """Tiny ImageNet dataset."""
    
    def __init__(self, root, train=True, transform=None, download=False):
        self.root = root
        self.train = train
        self.transform = transform
        self.download = download
        self.classes = list(range(200))

        if download:
            self._download()

        if train:
            path = os.path.join(root, 'tiny-imagenet-200', 'train')
        else:
            path = os.path.join(root, 'tiny-imagenet-200', 'val')

        super(TinyImageNetDataset, self).__init__(path, transform=transform)

    def _download(self):
        dataset_folder = os.path.join(self.root, 'tiny-imagenet-200')
        if os.path.exists(dataset_folder):
            print("Dataset already exists. No need to download.")
            return

        if not os.path.exists(self.root):
            os.makedirs(self.root)

        url = 'http://cs231n.stanford.edu/tiny-imagenet-200.zip'
        file_path = os.path.join(self.root, 'tiny-imagenet-200.zip')
        urllib.request.urlretrieve(url, file_path)

        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(self.root)

        os.remove(file_path)
    

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
    'TinyImageNet': TinyImageNetDataset,
}

def get_transforms(dataset_name):

    mean, std = MEAN[dataset_name], STD[dataset_name]
    size = SIZE[dataset_name]
    
    normalize = transforms.Normalize(mean=mean, std=std)

    train_transform = transforms.Compose([
            transforms.Resize(size=size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
    ])
    
    test_transform = transforms.Compose([
            transforms.Resize(size=size),
            transforms.ToTensor(),
            normalize,
    ])
    
    return train_transform, test_transform


def get_dataset(dataset_name, root, train=True, transform=None, download=True):
    if dataset_name == 'TinyImageNet':
        return TinyImageNetDataset(root, train=train, transform=transform, download=download)
    elif dataset_name == 'cifar10':
        return CIFAR10(root, train=train, transform=transform, download=download)
    elif dataset_name == 'cifar100':
        return CIFAR100(root, train=train, transform=transform, download=download)
    else:
        raise ValueError('Unknown dataset name: {}'.format(dataset_name))