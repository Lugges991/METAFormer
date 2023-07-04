import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset



class SingleAtlas(Dataset):
    def __init__(self, df, augment=0.0):
        self.df = df
        self.augment = augment

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        x = torch.tensor(np.loadtxt(self.df.iloc[idx].aal)).float()

        if np.random.rand() > self.augment:
            # add noise
            x += torch.randn_like(x) * 0.01

        # z_score normalization
        x = (x - x.mean()) / x.std()

        label = torch.tensor([self.df.iloc[idx].LABELS])
        label = torch.eye(2)[label].float().squeeze()

        return x, label

class MultiAtlas(Dataset):
    def __init__(self, df, augment=0.0):
        self.df = df
        self.augment = augment

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        aal = torch.tensor(np.loadtxt(self.df.iloc[idx].aal)).float()
        cc200 = torch.tensor(np.loadtxt(self.df.iloc[idx].cc200)).float()
        do160 = torch.tensor(np.loadtxt(
            self.df.iloc[idx].dosenbach160)).float()

        if np.random.rand() > self.augment:
            # add noise
            aal += torch.randn_like(aal) * 0.01
            cc200 += torch.randn_like(cc200) * 0.01
            do160 += torch.randn_like(do160) * 0.01

        # z_score normalization
        aal = (aal - aal.mean()) / aal.std()
        cc200 = (cc200 - cc200.mean()) / cc200.std()
        do160 = (do160 - do160.mean()) / do160.std()

        label = torch.tensor([self.df.iloc[idx].LABELS])
        label = torch.eye(2)[label].float().squeeze()

        return (aal, cc200, do160), label


class ImputationDataset(Dataset):
    # TODO: will be added soon
    pass
