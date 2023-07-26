import copy
import numpy as np
import torchvision.datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

def build_dataloader(
        trainset,
        testset,
        num_meta_total=1000,
        batch_size=100,
):

    train_dataloader_unshuffled = DataLoader(trainset, batch_size=batch_size, shuffle=False, pin_memory=True)

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

    return train_dataloader, meta_dataloader, test_dataloader, train_dataloader_unshuffled
