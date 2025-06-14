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
        # Daten mit xarray laden
        self.data = xr.open_dataset(nc_file_path)
        cftime_times = self.data['time'].values
        times_iso = [t.isoformat() for t in cftime_times]
        # jetzt echte numpy-datetime64[ns]
        pd_times = pd.to_datetime(times_iso)
        # als neue Koordinate setzen (oder einfach separat speichern)
        self.data = self.data.assign_coords(time=pd_times)
        
        # Statistik berechnen
        self.min_value = self.data['MSL'].min().values
        self.max_value = self.data['MSL'].max().values
        self.mean_value = self.data['MSL'].mean(dim='time').values
        self.std_value = self.data['MSL'].std(dim='time').values
        self.sqrt = sqrt

        # Entferne Polwerte
        if drop_pol:
            self.data['MSL'] = self.data['MSL'].where(self.data['lat'] < 88.5, 0)
        self.normalize_data()

    def load_or_calculate_stats(self):
        if os.path.exists(self.stats_file_path):
            with open(self.stats_file_path, 'r') as f:
                stats = json.load(f)
                self.mean_value = np.array(stats['mean_value'])
                self.std_value = np.array(stats['std_value'])
        else:
            self.mean_value = self.data['MSL'].mean(dim='time').compute().values
            self.std_value = self.data['MSL'].std(dim='time').compute().values
            stats = {
                'mean_value': self.mean_value.tolist(),
                'std_value': self.std_value.tolist()
            }
            with open(self.stats_file_path, 'w') as f:
                json.dump(stats, f)

    def normalize_data(self):
        self.data['MSL'] = (self.data['MSL'] - self.mean_value) / self.std_value
        lat = self.data['lat'].values
        if self.sqrt:
            lat_cos = np.sqrt(np.abs(np.cos(np.deg2rad(lat))))
        else:
            lat_cos = np.abs(np.cos(np.deg2rad(lat)))
        
        lat_cos = np.where(lat >= 88.5, 1, lat_cos)
        self.data['MSL'] *= lat_cos[:, None]

    def renormalize_data(self, normalized_data):
        lat = self.data['lat'].values
        if self.sqrt:
            lat_cos = np.sqrt(np.abs(np.cos(np.deg2rad(lat))))
        else:
            lat_cos = np.abs(np.cos(np.deg2rad(lat)))
        lat_cos = np.where(lat >= 88.5, 1, lat_cos)
        return normalized_data / lat_cos[:, None] * self.std_value + self.mean_value

    def __len__(self):
        return len(self.data['time'])

    def __getitem__(self, idx):
        time_step_data = self.data.isel(time=idx).compute()
        spatial_data = time_step_data['MSL'].values
        time = self.data.time.values[idx]
        
        return torch.FloatTensor(spatial_data).unsqueeze(0), str(time)







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
    