from torch.utils.data import Dataset, DataLoader
from pathlib import Path

import numpy as np

from av2.datasets.motion_forecasting import scenario_serialization
from av2.datasets.motion_forecasting.data_schema import ArgoverseScenario, ObjectState, ObjectType, Track, TrackCategory


class MotionForecastingDataset(Dataset):
    def __init__(self, dataset_dir):
        self.parquet_dirs = sorted(Path(dataset_dir).rglob("*.parquet"))

    def __getitem__(self, idx):
        scenario = scenario_serialization.load_argoverse_scenario_parquet(self.parquet_dirs[idx])
        return scenario

    def __len__(self):
        return len(self.parquet_dirs)