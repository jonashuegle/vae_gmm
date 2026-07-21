import torch
from torch.utils.data import Dataset
from torch.utils.data import random_split
import xarray as xr
import numpy as np
import pandas as pd

import pytorch_lightning as pl
from torch.utils.data import DataLoader


class CustomDataset(Dataset):
    """
    Custom Dataset for loading and processing MSL data from a NetCDF file and NAM data from a CSV file. The dataset normalizes the MSL data and provides a method to retrieve a 60-day period of data.

    Attributes:
        data (xarray.Dataset): Loaded NetCDF dataset containing MSL data.
        times (pd.DatetimeIndex): List of all time points from the NetCDF dataset.
        min_value (float): Minimum value of MSL in the dataset.
        max_value (float): Maximum value of MSL in the dataset.
        mean_value (float): Mean value of MSL in the dataset.
        std_value (float): Standard deviation of MSL in the dataset.
        sqrt (bool): Whether to apply square root transformation to latitude values.
    Parameters:
        nc_file_path (str): Path to the NetCDF file containing MSL data.
        drop_pol (bool): Whether to drop polar values from the MSL data. Defaults to True.
        sqrt (bool): Whether to apply square root transformation to latitude values. Defaults to False
    """
    def __init__(self, nc_file_path, drop_pol=True, sqrt=False, save_ram=True):
        # Load MSL data from NetCDF file (handling of no leap date time format because of the missing 29.02.xxxx)
        self.data = xr.open_dataset(nc_file_path)
        cftime_times = self.data['time'].values
        times_iso = [t.isoformat() for t in cftime_times]
        pd_times = pd.to_datetime(times_iso)
        self.data = self.data.assign_coords(time=pd_times)
        self.sqrt = sqrt

        # Remove polar values
        if drop_pol:
            self.data['MSL'] = self.data['MSL'].where(self.data['lat'] < 88.5, 0)

        # Calculate statistics
        self.mean_value = self.data['MSL'].mean().values
        self.std_value = self.data['MSL'].std().values

        # Normalize the data
        self.normalize_data()
        if np.isnan(self.data['MSL']).any():
            print("Warnung: NaNs nach Normalisierung!")

        # Prepare the dataset for training
        # Create a list of all spatial data and corresponding times 
        # Save data in RAM to avoid bottlenecks during training
        self.all_spatial_data = []
        self.all_times = []
        for i in range(len(self.data['time'])):
            arr = self.data['MSL'].isel(time=i).values
            arr = torch.FloatTensor(arr).unsqueeze(0)  # shape: [1, ...]
            self.all_spatial_data.append(arr)
            self.all_times.append(str(self.data.time.values[i]))

        # Optional: Remove xarray data to save RAM
        if save_ram:
            self.data.close()
            del self.data
            self.data = None

    def normalize_data(self):
        """
        Normalizes the MSL data by subtracting the mean and dividing by the standard deviation. Then, it applies a cosine transformation based on latitude values.
        If the square root transformation is enabled, it applies the square root of the absolute cosine of the latitude.
        If the latitude is greater than or equal to 88.5, it sets the value to 1.
        """
        self.std_value[self.std_value == 0] = 1.0
        self.data['MSL'] = (self.data['MSL'] - self.mean_value) / self.std_value
        lat = self.data['lat'].values
        if self.sqrt:
            lat_cos = np.sqrt(np.abs(np.cos(np.deg2rad(lat))))
        else:
            lat_cos = np.abs(np.cos(np.deg2rad(lat)))
        lat_cos = np.where(lat >= 88.5, 1, lat_cos)
        self.data['MSL'] *= lat_cos[:, None]

    def __len__(self):
        """
        Returns the total number of spatial data points in the dataset.
        """
        return len(self.all_spatial_data)

    def __getitem__(self, idx):
        """
        Retrieves the spatial data and corresponding time for a given index.
        Parameters:
            idx (int): Index of the data point to retrieve.
        Returns:
            tuple: A tuple containing the spatial data and the corresponding time as a string.
        """
        return self.all_spatial_data[idx], self.all_times[idx]





class DataModule(pl.LightningDataModule):
    """
    Pytorch Lightning DataModule for loading and processing the CustomDataset. It handles the preparation of training, validation, and test datasets, and provides data loaders for each.

    Attributes:
        data_dir (str): Directory containing the dataset files.
        batch_size (int): Batch size for the data loaders.
        num_workers (int): Number of worker threads for data loading.
        shuffle_train (bool): Flag to control shuffling of the training dataloader (used by callbacks).
    Parameters:
        data_dir (str): Directory containing the dataset files.
        batch_size (int): Batch size for the data loaders.
        num_workers (int): Number of worker threads for data loading.
    """
    def __init__(self, data_dir, batch_size, num_workers):
        super(DataModule, self).__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

        # will be toggled by SwitchShuffleCallback
        self.shuffle_train = True

        # placeholders
        self.full_dataset = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def prepare_data(self, drop_pol=True):
        """
        Optional data preparation step (e.g. download). Not used here.
        """
        pass

    def setup(self, stage=None):
        """
        Sets up the dataset for training, validation, and testing.
        Lightning calls this with an optional 'stage' argument.

        Args:
            stage (str, optional): One of "fit", "validate", "test" or None.
        """
        # load the full dataset only once
        if self.full_dataset is None:
            self.full_dataset = CustomDataset(
                self.data_dir,
                drop_pol=True,
                sqrt=False,
                save_ram=True
            )

        # create train/val split for fitting
        if stage == "fit" or stage is None:
            train_size = int(0.8 * len(self.full_dataset))
            val_size = len(self.full_dataset) - train_size

            self.train_dataset, self.val_dataset = random_split(
                self.full_dataset,
                [train_size, val_size],
                generator=torch.Generator().manual_seed(42)
            )

        # make sure we have something for test_dataloader
        if stage == "test" or stage is None:
            if self.test_dataset is None:
                self.test_dataset = self.full_dataset

    def train_dataloader(self):
        """
        Returns a DataLoader for the training dataset.
        This DataLoader can be configured to shuffle via self.shuffle_train.
        """
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=self.shuffle_train
        )

    def val_dataloader(self):
        """
        Returns a DataLoader for the validation dataset.
        This DataLoader does not shuffle the data.
        """
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False
        )

    def test_dataloader(self):
        """
        Returns a DataLoader for the test dataset.
        This DataLoader does not shuffle the data.
        """
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False
        )

    def all_data_dataloader(self):
        """
        Returns a DataLoader for the full dataset.
        This DataLoader does not shuffle the data.
        """
        return DataLoader(
            self.full_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False
        )
