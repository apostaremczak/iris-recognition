import cv2
from torch.utils.data import Dataset


class IrisDataset(Dataset):
    """
    PyTorch dataset for storing users' iris pictures.
    """
    def __init__(self, irides_dir: str, transform=None):
        """
        :param irides_dir:
        :param transform:
        """
        super().__init__()
        self.irides_dir = irides_dir
        self.transform = transform

    def __len__(self):
        return 0

    def __getitem__(self, idx):
        return None
