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
            x += torch.randn_like(x) * 0.01

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
            aal += torch.randn_like(aal) * 0.01
            cc200 += torch.randn_like(cc200) * 0.01
            do160 += torch.randn_like(do160) * 0.01

        aal = (aal - aal.mean()) / aal.std()
        cc200 = (cc200 - cc200.mean()) / cc200.std()
        do160 = (do160 - do160.mean()) / do160.std()

        label = torch.tensor([self.df.iloc[idx].LABELS])
        label = torch.eye(2)[label].float().squeeze()

        return (aal, cc200, do160), label


class ImputationDataset(Dataset):
    def __init__(self, df, mask_ratio=0.0):
        self.df = df
        self.mask_ratio = mask_ratio

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        aal = torch.tensor(np.loadtxt(self.df.iloc[idx].aal)).float()
        cc200 = torch.tensor(np.loadtxt(self.df.iloc[idx].cc200)).float()
        do160 = torch.tensor(np.loadtxt(
            self.df.iloc[idx].dosenbach160)).float()

        aal = (aal - aal.mean()) / aal.std()
        cc200 = (cc200 - cc200.mean()) / cc200.std()
        do160 = (do160 - do160.mean()) / do160.std()

        # generate mask where self.mask_ratio is 1 else 0
        aal_mask = torch.tensor(np.random.choice(
            [0, 1], size=aal.shape, p=[1-self.mask_ratio, self.mask_ratio]))
        cc200_mask = torch.tensor(np.random.choice(
            [0, 1], size=cc200.shape, p=[1-self.mask_ratio, self.mask_ratio]))
        do160_mask = torch.tensor(np.random.choice(
            [0, 1], size=do160.shape, p=[1-self.mask_ratio, self.mask_ratio]))

        # note that torch multihead attention requires mask to be 0 where attention is applied
        aal_masked = aal * ~aal_mask
        cc200_masked = cc200 * ~cc200_mask
        do160_masked = do160 * ~do160_mask

        return (aal, cc200, do160), (aal_masked, cc200_masked, do160_masked), (aal_mask, cc200_mask, do160_mask)
