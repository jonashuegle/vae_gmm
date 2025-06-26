import torch
from torch.utils.data import Dataset
from torch.utils.data import random_split
import xarray as xr
import numpy as np
import pandas as pd

import pytorch_lightning as pl
from torch.utils.data import DataLoader

import os
import json


class CustomDataset(Dataset):
    def __init__(self, nc_file_path, drop_pol=True, sqrt=False):
        self.data = xr.open_dataset(nc_file_path)
        cftime_times = self.data['time'].values
        times_iso = [t.isoformat() for t in cftime_times]
        pd_times = pd.to_datetime(times_iso)
        self.data = self.data.assign_coords(time=pd_times)
        self.sqrt = sqrt

        # Pol entfernen und normalisieren wie vorher
        if drop_pol:
            self.data['MSL'] = self.data['MSL'].where(self.data['lat'] < 88.5, 0)

        self.mean_value = self.data['MSL'].mean(dim='time').values
        self.std_value = self.data['MSL'].std(dim='time').values
        print("Anzahl std_value == 0:", np.sum(self.std_value == 0))
        print("std_value min/max:", np.min(self.std_value), np.max(self.std_value))
        self.normalize_data()
        if np.isnan(self.data['MSL']).any():
            print("Warnung: NaNs nach Normalisierung!")

        # **HIER: ALLE Zeitschritte in RAM laden**
        # -> Liste von Torch-Tensors, damit __getitem__ nur noch RAM-Zugriff ist
        self.all_spatial_data = []
        self.all_times = []
        for i in range(len(self.data['time'])):
            arr = self.data['MSL'].isel(time=i).values
            arr = torch.FloatTensor(arr).unsqueeze(0)  # shape: [1, ...]
            self.all_spatial_data.append(arr)
            self.all_times.append(str(self.data.time.values[i]))

        # Optional: Löse xarray wieder auf, um RAM zu sparen
        del self.data

    def normalize_data(self):
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
        return len(self.all_spatial_data)

    def __getitem__(self, idx):
        return self.all_spatial_data[idx], self.all_times[idx]





class DataModule(pl.LightningDataModule):
    
    def __init__(self, data_dir, batch_size, num_workers):
        super(DataModule, self).__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data = None
    
    def prepare_data(self, drop_pol = True):
        pass

    def setup(self, stage = None):
                
        self.full_dataset = CustomDataset(self.data_dir)
        train_size = int(0.8 * len(self.full_dataset))
        val_size = len(self.full_dataset) - train_size

        self.train_dataset, self.val_dataset = random_split(
            self.full_dataset, [train_size, val_size], generator=torch.Generator()
        )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)
    
    def all_data_dataloader(self):
        return DataLoader(self.full_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)
    