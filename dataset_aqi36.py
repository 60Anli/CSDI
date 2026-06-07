import os
import pickle

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


def _resolve_file(candidates):
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError("None of these files exist: " + ", ".join(candidates))


class AQI36_Dataset(Dataset):
    def __init__(
        self,
        eval_length=36,
        target_dim=36,
        mode="train",
        val_len=0.1,
        mask_sensor=None,
    ):
        self.eval_length = eval_length
        self.target_dim = target_dim
        self.mode = mode
        self.mask_sensor = mask_sensor or []

        meanstd_path = _resolve_file(
            [
                "./data/pm25/pm25_meanstd.pk",
                "./PriSTI-main/data/pm25/pm25_meanstd.pk",
            ]
        )
        with open(meanstd_path, "rb") as f:
            self.train_mean, self.train_std = pickle.load(f)

        if mode == "train":
            month_list = [1, 2, 4, 5, 7, 8, 10, 11]
            flag_for_histmask = [0, 1, 0, 1, 0, 1, 0, 1]
        elif mode == "valid":
            month_list = [2, 5, 8, 11]
            flag_for_histmask = None
        elif mode == "test":
            month_list = [3, 6, 9, 12]
            flag_for_histmask = None
        else:
            raise ValueError(f"Unknown mode: {mode}")
        self.month_list = month_list

        ground_path = _resolve_file(
            [
                "./data/pm25/Code/STMVL/SampleData/pm25_ground.txt",
                "./data/pm25/SampleData/pm25_ground.txt",
                "./PriSTI-main/data/pm25/SampleData/pm25_ground.txt",
            ]
        )
        missing_path = _resolve_file(
            [
                "./data/pm25/Code/STMVL/SampleData/pm25_missing.txt",
                "./data/pm25/SampleData/pm25_missing.txt",
                "./PriSTI-main/data/pm25/SampleData/pm25_missing.txt",
            ]
        )

        df = pd.read_csv(ground_path, index_col="datetime", parse_dates=True)
        df_gt = pd.read_csv(missing_path, index_col="datetime", parse_dates=True)

        self.observed_data = []
        self.observed_mask = []
        self.gt_mask = []
        self.index_month = []
        self.position_in_month = []
        self.valid_for_histmask = []
        self.use_index = []
        self.cut_length = []

        for i, month in enumerate(month_list):
            current_df = df[df.index.month == month]
            current_df_gt = df_gt[df_gt.index.month == month]
            if mode == "train" and month in [2, 5, 8, 11]:
                cut_len = int(val_len * len(current_df))
                if cut_len > 0:
                    current_df = current_df[:-cut_len]
                    current_df_gt = current_df_gt[:-cut_len]
            if mode == "valid":
                cut_len = int(val_len * len(current_df))
                if cut_len > 0:
                    current_df = current_df[-cut_len:]
                    current_df_gt = current_df_gt[-cut_len:]

            current_length = len(current_df) - eval_length + 1
            last_index = len(self.index_month)
            self.index_month += np.array([i] * current_length).tolist()
            self.position_in_month += np.arange(current_length).tolist()
            if mode == "train":
                self.valid_for_histmask += np.array(
                    [flag_for_histmask[i]] * current_length
                ).tolist()

            c_mask = 1 - current_df.isnull().values
            c_gt_mask = 1 - current_df_gt.isnull().values
            for sensor in self.mask_sensor:
                c_gt_mask[:, sensor] = 0
                if mode == "train":
                    c_mask[:, sensor] = 0

            c_data = (
                (current_df.fillna(0).values - self.train_mean) / self.train_std
            ) * c_mask
            self.observed_mask.append(c_mask)
            self.gt_mask.append(c_gt_mask)
            self.observed_data.append(c_data)

            if mode == "test":
                n_sample = len(current_df) // eval_length
                c_index = np.arange(
                    last_index, last_index + eval_length * n_sample, eval_length
                )
                self.use_index += c_index.tolist()
                self.cut_length += [0] * len(c_index)
                if len(current_df) % eval_length != 0:
                    self.use_index += [len(self.index_month) - 1]
                    self.cut_length += [eval_length - len(current_df) % eval_length]

        if mode != "test":
            self.use_index = np.arange(len(self.index_month))
            self.cut_length = [0] * len(self.use_index)

        if mode == "train":
            ind = -1
            self.index_month_histmask = []
            self.position_in_month_histmask = []
            for _ in range(len(self.index_month)):
                while True:
                    ind += 1
                    if ind == len(self.index_month):
                        ind = 0
                    if self.valid_for_histmask[ind] == 1:
                        self.index_month_histmask.append(self.index_month[ind])
                        self.position_in_month_histmask.append(
                            self.position_in_month[ind]
                        )
                        break
        else:
            self.index_month_histmask = self.index_month
            self.position_in_month_histmask = self.position_in_month

    def __getitem__(self, org_index):
        index = self.use_index[org_index]
        c_month = self.index_month[index]
        c_index = self.position_in_month[index]

        hist_source = np.random.randint(0, len(self.use_index))
        hist_month = self.index_month_histmask[hist_source]
        hist_index = self.position_in_month_histmask[hist_source]

        return {
            "observed_data": self.observed_data[c_month][
                c_index : c_index + self.eval_length
            ],
            "observed_mask": self.observed_mask[c_month][
                c_index : c_index + self.eval_length
            ],
            "gt_mask": self.gt_mask[c_month][c_index : c_index + self.eval_length],
            "hist_mask": self.observed_mask[hist_month][
                hist_index : hist_index + self.eval_length
            ],
            "timepoints": np.arange(self.eval_length),
            "cut_length": self.cut_length[org_index],
        }

    def __len__(self):
        return len(self.use_index)


def get_dataloader(
    batch_size,
    device,
    val_len=0.1,
    num_workers=1,
    seed=42,
    mask_sensor=None,
):
    generator = torch.Generator()
    generator.manual_seed(seed)

    def worker_init_fn(worker_id):
        np.random.seed(seed + worker_id)

    train_dataset = AQI36_Dataset(
        mode="train", val_len=val_len, mask_sensor=mask_sensor
    )
    valid_dataset = AQI36_Dataset(
        mode="valid", val_len=val_len, mask_sensor=mask_sensor
    )
    test_dataset = AQI36_Dataset(mode="test", val_len=val_len, mask_sensor=mask_sensor)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        generator=generator,
        worker_init_fn=worker_init_fn,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        generator=generator,
        worker_init_fn=worker_init_fn,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        generator=generator,
        worker_init_fn=worker_init_fn,
    )

    scaler = torch.from_numpy(train_dataset.train_std).to(device).float()
    mean_scaler = torch.from_numpy(train_dataset.train_mean).to(device).float()
    return train_loader, valid_loader, test_loader, scaler, mean_scaler
