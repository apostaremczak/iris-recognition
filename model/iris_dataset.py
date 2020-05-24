import numpy as np
import torch
from glob import glob
from skimage import io
from torch.utils.data import Dataset

from utils.preprocessing import extract_user_sample_ids


class IrisDataset(Dataset):
    """
    PyTorch dataset for storing users' iris pictures.
    """

    def __init__(self, normalized_irides_dir: str, transform=None):
        """
        :param normalized_irides_dir:
        :param transform:
        """
        super().__init__()
        self.irides_dir = normalized_irides_dir
        self.transform = transform
        self.file_paths = sorted(glob(normalized_irides_dir + "/*"))

    def __len__(self):
        """
        Size of the dataset
        """
        return len(self.file_paths)

    def __getitem__(self, idx):
        """
        Get i-th sample
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        filename = self.file_paths[idx]
        image = io.imread(filename)
        user_id, _ = extract_user_sample_ids(filename)

        sample = {
            "image": image,
            "user_id": np.array([user_id]).astype("float")
        }

        if self.transform:
            sample = self.transform(sample)

        return sample
