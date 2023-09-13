from torchvision import datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from consts import DATASET_CIFAR10, DATASET_CIFAR100, DATASET_IMAGENET, DATASET_IMAGENET_REAL, DATASET_OXFORD102, DATASET_OXFORDIIIT
import os
from dataset_loaders.ImageNet21k import ImageNet21k
from dataset_loaders.ImageNet1k import ImageNet1k
from dataset_loaders.datasset_config import DatasetConfig

def get_loader(config:DatasetConfig, is_train:bool) -> DataLoader:
    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    if config.get_name() == DATASET_CIFAR10:
        dataset = datasets.CIFAR10(root=config.get_dataset_dir(), 
                                   train=is_train, 
                                   transform=transform_test if not is_train else transform_train)
    elif config.get_name() == DATASET_CIFAR100:
        dataset = datasets.CIFAR100(root=config.get_dataset_dir(), 
                                    train=is_train, 
                                    transform=transform_test if not is_train else transform_train)
    elif config.get_name() == DATASET_IMAGENET:
        dataset = ImageNet21k(root=config.get_dataset_dir(),
                              is_train=is_train,
                              config=config,
                              transform=transform_test if not is_train else transform_train)
    elif config.get_name() == DATASET_IMAGENET_REAL:
        dataset = ImageNet1k(root=config.get_dataset_dir(),
                             config=config,
                             is_train=is_train,
                             transform=transform_test if not is_train else transform_train)
    elif config.get_name() == DATASET_OXFORD102:
        dataset = datasets.Flowers102(root=config.get_dataset_dir(),
                                      split='train' if is_train else 'test', 
                                      transform=transform_test if not is_train else transform_train,
                                      download=True)
        
    elif config.get_name() == DATASET_OXFORDIIIT:
        dataset = datasets.OxfordIIITPet(root=config.get_dataset_dir(),
                                         split='trainval' if is_train else 'test', 
                                         transform=transform_test if not is_train else transform_train,
                                         download=True)

    return DataLoader(dataset, batch_size=32 if is_train else config.get_batch_size(), num_workers=2, pin_memory=True)    