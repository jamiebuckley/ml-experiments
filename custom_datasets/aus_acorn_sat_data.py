from torch.utils.data import Dataset
import logging
import custom_datasets.aus_acorn_data_retriever as data_retriever
import pandas as pd
import torch
import os
from config import ROOT_DIR
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

class AusAcornSatData(Dataset):

    csv_sat_path = os.path.join(ROOT_DIR, 'custom_datasets', 'aus_acorn_sites.csv')
    raw_acorn_sat_path = os.path.join(ROOT_DIR, '.rawdata', 'acorn_sat')

    min_values = {}
    max_values = {}

    def __init__(self, mode, train=True, subset=0):
        self.mode = mode
        self.parquet_path = os.path.join(self.raw_acorn_sat_path, self.mode + '.parquet')
        if not os.path.exists(self.parquet_path):
            data_retriever.AusAcornSatDataRetriever().download(mode=mode)

        df = pd.read_parquet(self.parquet_path)

        # Reduce data set size to run on my terrible GPU
        if subset > 0:
            df = df.iloc[::subset]

        self.cols = [mode, 'lat', 'lon', 'elevation', 'days_since_start', 'days_ssoy_sin', 'days_ssoy_cos']
        self.feature_cols = [item for item in self.cols if item != mode]

        for col in self.cols:
            self.min_values[col] = c_min = df[col].min()
            self.max_values[col] = c_max = df[col].max()
            df[col] = (df[col] - c_min) / (c_max - c_min)

        df_train, df_val = train_test_split(df, test_size=0.2, random_state=42)

        if train:
            df = df_train
        else:
           df = df_val

        self.df = df
        self.features = df[self.feature_cols].values.astype('float32')
        self.targets = df[mode].values.astype('float32')

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        x = torch.tensor(self.features[idx])
        y = torch.tensor(self.targets[idx], dtype=torch.float32).unsqueeze(0)
        return x, y

    def normalize(self, values):
        norm = {}
        for col in self.feature_cols:
            c_min = self.min_values[col]
            c_max = self.max_values[col]
            norm[col] = (values[col] - c_min) / (c_max - c_min)

        # todo, improve this for feature col tweaking
        x = torch.tensor([
            norm['lat'],
            norm['lon'],
            norm['elevation'],
            norm['days_since_start'],
            values['days_ssoy_sin'],
            values['days_ssoy_cos']
        ], dtype=torch.float32)
        return x

    def denormalize(self, value):
        c_min = self.min_values[self.mode]
        c_max = self.max_values[self.mode]
        return value * (c_max - c_min) + c_min

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test = AusAcornSatData(mode="max")
