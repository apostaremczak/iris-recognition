from dataclasses import dataclass
from typing import Dict

import os
import torch
import torch.nn as nn

from torch.optim import Adam
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

from model.iris_classifier_model import IrisClassifier, DATA_TRANSFORMS

# Directory with normalized photos splitted into train and val subsets
INPUT_DATA_DIR = 'data/tmp/normalized_splitted'


@dataclass
class TrainConfig:
    data: Dict[str, ImageFolder]
    loaders: Dict[str, DataLoader]
    criterion: nn.Module = nn.CrossEntropyLoss()
    learning_rate: float = 0.0002
    num_epochs: int = 100

    def __post_init__(self):
        self.class_names = sorted(self.data['train'].classes)
        self.data_sizes = {x: len(self.data[x]) for x in ['train', 'val']}
        self.device = torch.device("cuda:0" if torch.cuda.is_available()
                                   else "cpu")
        self.model = IrisClassifier(class_names=self.class_names,
                                    num_classes=len(self.class_names),
                                    load_from_checkpoint=False)
        self.optimizer = Adam(
            params=self.model.parameters(),
            lr=self.learning_rate
        )


def create_train_config(batch_size: int = 16,
                        shuffle: bool = True,
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
