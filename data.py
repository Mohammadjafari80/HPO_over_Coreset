import torch
from torch.utils.data import Dataset
import os
import zipfile
import urllib.request
from torchvision.datasets import ImageFolder
from torchvision.datasets import CIFAR10, CIFAR100


class CustomWeightedDataset(Dataset):
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