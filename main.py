import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from METAFormer.dataloader import ImputationDataset, MultiAtlas
from METAFormer.models import METAFormer, METAWrapper
from METAFormer.utils import pretrain, train, test


cfg = {
    "BATCH_SIZE": 256,
    "LR": 1e-4,
    "VAL_AFTER": 1,
    "LOSS": nn.BCEWithLogitsLoss(),
    "WEIGHT_DECAY": 0.00,
    "DROP": 0.0,
    "AUG": 0.0,
    "GAMMA": 0.9,
    "DEVICE": "cuda:0",
    "PATIENCE": 20,
    "EPOCHS": 750,
}


def pretrain_train_cross_validate(args):

    device = cfg["DEVICE"]

    df = pd.read_csv(args.csv)

    # cross validation
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    y = df.LABELS
    x = df.drop("LABELS", axis=1)
    accs = []

    table_cols = ['Fold', 'Accuracy', 'Precision', 'Recall',
                  'F1', 'AUC', 'AP', 'FPR', 'FNR', 'TPR', 'TNR']

    vals = []

    for fold, (train_idx, test_idx) in enumerate(kfold.split(x, y)):
        print(80 * "=")
        print(f"Fold {fold}")
        print(80 * "=")

        train_df, val_df = train_test_split(
            df.iloc[train_idx], test_size=0.3, random_state=42)
        test_df = df.iloc[test_idx]

        train_loader = DataLoader(MultiAtlas(
            train_df, softsign=False), batch_size=cfg["BATCH_SIZE"], shuffle=True, num_workers=8, pin_memory=True)
        val_loader = DataLoader(MultiAtlas(
            val_df, softsign=False), batch_size=cfg["BATCH_SIZE"], shuffle=False, num_workers=8, pin_memory=True)
        test_loader = DataLoader(MultiAtlas(
            test_df, softsign=False), batch_size=cfg["BATCH_SIZE"], shuffle=False, num_workers=8, pin_memory=True)

        # Pretrain
        pretrain_loader = DataLoader(ImputationDataset(
            train_df), batch_size=cfg["BATCH_SIZE"], shuffle=True, num_workers=8, pin_memory=True)
        preval_loader = DataLoader(ImputationDataset(
            val_df), batch_size=cfg["BATCH_SIZE"], shuffle=False, num_workers=8, pin_memory=True)

        pt_model = METAWrapper(d_model=256, dim_feedforward=128, num_encoder_layers=2,
                               num_heads=4, dropout=cfg["DROP"]).to(cfg["DEVICE"])

        pt_optim = optim.AdamW(METAWrapper.parameters(),
                               lr=cfg["LR"], weight_decay=cfg["WEIGHT_DECAY"])

        pretrained = pretrain(model=METAWrapper, train_loader=pretrain_loader, val_loader=preval_loader,
                              optimizer=pt_optim, epochs=1000, device=device, stage=f"pretrain fold {fold}", patience=50, scheduler=None)

        print("Pretraining finished...")

        # Load pretrained weights
        model = METAFormer(d_model=256, dim_feedforward=128, num_encoder_layers=2,
                           num_heads=4, dropout=cfg["DROP"]).to(cfg["DEVICE"])
        model.aal_encoder.load_state_dict(
                pretrained.aal_encoder.state_dict())
        model.cc200_encoder.load_state_dict(
            pretrained.cc200_encoder.state_dict())
        model.dos160_encoder.load_state_dict(
            pretrained.dos160_encoder.state_dict())

        optimizer = optim.AdamW(
            model.parameters(), lr=cfg["LR"], weight_decay=cfg["WEIGHT_DECAY"])

        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
        criterion = nn.BCEWithLogitsLoss().to(cfg["DEVICE"])

        trained_model, best_a = train(model, train_loader, val_loader, criterion, optimizer,
                                      cfg["EPOCHS"], device, patience=cfg["PATIENCE"], scheduler=scheduler, return_best_acc=True)

        # Test
        print("Testing...")
        trues, preds = test(trained_model, test_loader, device)

        acc = accuracy_score(trues, preds)
        accs.append(acc)
        cm = confusion_matrix(trues, preds)
        precision = cm[1, 1] / (cm[1, 1] + cm[0, 1])
        recall = cm[1, 1] / (cm[1, 1] + cm[1, 0])
        f1 = 2 * (precision * recall) / (precision + recall)
        auc = roc_auc_score(trues, preds)
        ap = average_precision_score(trues, preds)
        fpr = cm[0, 1] / (cm[0, 1] + cm[0, 0])
        fnr = cm[1, 0] / (cm[1, 0] + cm[1, 1])
        tpr = cm[1, 1] / (cm[1, 1] + cm[1, 0])
        tnr = cm[0, 0] / (cm[0, 0] + cm[0, 1])

        print(f"Fold {fold}: Accuracy: {acc}")
        print(f"Fold {fold}: Confusion matrix:\n{cm}")

        vals.append([fold, acc, precision, recall,
                     f1, auc, ap, fpr, fnr, tpr, tnr])

    results = pd.DataFrame(vals, columns=table_cols)
    print(results)
    print(80 * "=")
    print(f"Mean accuracy: {np.mean(accs)}")
    print(f"Std accuracy: {np.std(accs)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True)
    args = parser.parse_args()

    pretrain_train_cross_validate(args)
