from pathlib import Path
from torch.utils.data import Dataset
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torchvision.datasets.utils import download_url

import os
import torch
import numpy as np
import pandas as pd

class TitansoftDataset(Dataset):
    """
    TitansoftDataset by Zino
    """

    def __init__(self, root: str, dataset_name: str, train=True, random_state=None):
        super(Dataset, self).__init__()

        self.classes = [0, 1]

        if isinstance(root, torch._six.string_classes):
            root = os.path.expanduser(root)
        self.root = Path(root)
        self.dataset_name = dataset_name
        self.train = train  # training set or test set
        self.file_name = self.dataset_name
        self.data_file = self.root / self.file_name
        
        
        
        train_df = self.readFileToDf(self.data_file / "train_80.csv")
        test_df = self.readFileToDf(self.data_file / "valid_80.csv")

        X_train =  train_df.drop(['label_month5_payment'], axis=1)
        X_test = test_df.drop(['label_month5_payment'], axis=1)
        y_train = train_df['label_month5_payment']
        y_test = test_df['label_month5_payment']


        # Standardize data (per feature Z-normalization, i.e. zero-mean and unit variance)
        scaler = StandardScaler().fit(X_train)
        X_train_stand = scaler.transform(X_train)
        X_test_stand = scaler.transform(X_test)

        # Scale to range [0,1]
        minmax_scaler = MinMaxScaler().fit(X_train_stand)
        X_train_scaled = minmax_scaler.transform(X_train_stand)
        X_test_scaled = minmax_scaler.transform(X_test_stand)

        if self.train:
            self.data = torch.tensor(X_train_scaled, dtype=torch.float32)
            self.targets = torch.tensor(y_train, dtype=torch.int64)
        else:
            self.data = torch.tensor(X_test_scaled, dtype=torch.float32)
            self.targets = torch.tensor(y_test, dtype=torch.int64)

        self.semi_targets = torch.zeros_like(self.targets)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target, semi_target, index)
        """
        sample, target, semi_target = self.data[index], int(self.targets[index]), int(self.semi_targets[index])

        return sample, target, semi_target, index

    def __len__(self):
        return len(self.data)

    def _check_exists(self):
        return os.path.exists(self.data_file)

    def download(self):
        """Download the ODDS dataset if it doesn't exist in root already."""

        if self._check_exists():
            return

        # download file
        download_url(self.urls[self.dataset_name], self.root, self.file_name)

        print('Done!')

    def readFileToDf(self, filepath):
        """ read file"""
        df = pd.read_csv(filepath)
        
        return df
