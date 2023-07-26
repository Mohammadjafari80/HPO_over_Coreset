import copy
import numpy as np
import torchvision.datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

def build_dataloader(
        seed=1,
        trainset,
        testset,
        num_meta_total=1000,
        batch_size=100,
):

    np.random.seed(seed)
    normalize = transforms.Normalize(
        mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
        std=[x / 255.0 for x in [63.0, 62.1, 66.7]],
    )

    train_transforms = transforms.Compose([
        transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    num_classes = len(trainset.classes)
    num_meta = int(num_meta_total / num_classes)

    index_to_meta = []
    index_to_train = []

    for class_index in range(num_classes):
        index_to_class = [index for index, label in enumerate(trainset.targets) if label == class_index]
        np.random.shuffle(index_to_class)
        index_to_meta.extend(index_to_class[:num_meta])
        index_to_class_for_train = index_to_class[num_meta:]
        
        index_to_train.extend(index_to_class_for_train)

    meta_dataset = copy.deepcopy(trainset)
    trainset.data = trainset.data[index_to_train]
    trainset.targets = list(np.array(trainset.targets)[index_to_train])
    meta_dataset.data = meta_dataset.data[index_to_meta]
    meta_dataset.targets = list(np.array(meta_dataset.targets)[index_to_meta])

    train_dataloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, pin_memory=True)
    meta_dataloader = DataLoader(meta_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    test_dataloader = DataLoader(testset, batch_size=batch_size, pin_memory=True)

    return train_dataloader, meta_dataloader, test_dataloader
