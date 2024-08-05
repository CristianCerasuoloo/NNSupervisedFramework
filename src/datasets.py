import os
import torch
import pandas as pd

from PIL import Image

from utils.logger import get_logger
from torch.utils.data import Dataset

logger = get_logger()

def default_loader(path):
    try:
        return Image.open(path).convert('RGB')
    except FileNotFoundError:
        logger.critical("File {} not found".format(path))
        return None

class ImageDataset(Dataset):
    """
    A dataset class that loads images and corresponding labels from a csv file.
    """
    def __init__(self, image_folder, label_csv, transform = None, loader=default_loader):
        super(ImageDataset, self).__init__()
        self.image_folder = image_folder
        self.transform = transform
        self.loader = loader

        self.labels=pd.read_csv(label_csv, header = None, sep=',')
        
    def __getitem__(self, index):
        """
        Given an index, returns the image and the labels of the sample at that index.

        Parameters
        ----------
        index : int
            Index of the sample

        Returns
        -------
        torch.Tensor, torch.Tensor
        """
        row=self.labels.iloc[index]

        path=os.path.join(self.image_folder, row[0])
        img = self.loader(path)
        if img is None:
            return None, None
        if self.transform is not None:
            img = self.transform(img)
        lab = torch.Tensor(row[1:].to_list())

        return img, lab
    
    def get_labels(self, index):
        """
        Returns the labels of the sample at index specified.

        Parameters
        ----------
        index : int
            Index of the sample

        Returns
        -------
        torch.Tensor
        """
        row = self.labels.iloc[index]
        return torch.Tensor(row[1:].to_list())
    
    def __len__(self):
        return len(self.labels[0])
