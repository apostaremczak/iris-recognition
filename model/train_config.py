from dataclasses import dataclass
from typing import Dict

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader


# Directory with normalized photos
INPUT_DATA_DIR = '../data/normalized_splitted'

# Data augmentation for training and validation
DATA_TRANSFORMS = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


@dataclass
class TrainConfig:
    data: Dict[str, ImageFolder]
    loaders: Dict[str, DataLoader]

    def __post_init__(self):
        self.class_names = self.data['train'].classes
        self.data_sizes = {x: len(self.data[x]) for x in ['train', 'val']}
        self.device = torch.device("cuda:0" if torch.cuda.is_available()
                                   else "cpu")

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD
        self.lr_scheduler = lr_scheduler.StepLR

    def get_model(self) -> nn.Module:
        model = models.resnet50(pretrained=True)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, len(self.class_names))

        return model.to(self.device)


def create_train_config(batch_size: int = 4, shuffle: bool = True,
                        num_workers: int = 4) -> TrainConfig:
    image_datasets = {
        x: ImageFolder(os.path.join(INPUT_DATA_DIR, x), DATA_TRANSFORMS[x])
        for x in ['train', 'val']
    }

    data_loaders = {
        x: DataLoader(image_datasets[x], batch_size=batch_size,
                      shuffle=shuffle, num_workers=num_workers)
        for x in ['train', 'val']
    }

    return TrainConfig(image_datasets, data_loaders)
