import copy
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import torch.nn.functional as F
import torch.nn as nn

from METAFormer.dataloader import ImputationDataset


class MaskedMSELoss(nn.Module):
    """ Masked MSE Loss
        from https://github.com/gzerveas/mvts_transformer/blob/3f2e378bc77d02e82a44671f20cf15bc7761671a/src/models/loss.py
    """

    def __init__(self, reduction: str = 'mean'):

        super().__init__()

        self.reduction = reduction
        self.mse_loss = nn.MSELoss(reduction=self.reduction)

    def forward(self,
                y_pred: torch.Tensor, y_true: torch.Tensor, mask: torch.BoolTensor) -> torch.Tensor:
        """Compute the loss between a target value and a prediction.

        Args:
            y_pred: Estimated values
            y_true: Target values
            mask: boolean tensor with 0s at places where values should be ignored and 1s where they should be considered

        Returns
        -------
        if reduction == 'none':
            (num_active,) Loss for each active batch element as a tensor with gradient attached.
        if reduction == 'mean':
            scalar mean loss over batch as a tensor with gradient attached.
        """

        # for this particular loss, one may also elementwise multiply y_pred and y_true with the inverted mask
        masked_pred = torch.masked_select(y_pred, mask)
        masked_true = torch.masked_select(y_true, mask)

        return self.mse_loss(masked_pred, masked_true)


def pretrain(model, train_loader, val_loader, optimizer, epochs, device, stage, patience=1, scheduler=None):
    """ Pretrain the model on the train_df and validate on val_df
    """
    early_stopping = True if patience else False
    losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_model = None
    counter = 0

    crit_aal = MaskedMSELoss()
    crit_cc200 = MaskedMSELoss()
    crit_dos160 = MaskedMSELoss()

    with tqdm(range(epochs), unit="epoch") as tepoch:
        for epoch in tepoch:
            tepoch.set_description(f"Epoch {epoch}")
            epoch_losses = []
            running_loss = 0.0
            model.train()
            for i, ((aal, cc200, dos160), (aal_masked, cc200_masked, dos160_masked), (aal_mask, cc200_mask, dos160_mask)) in enumerate(train_loader):
                aal, cc200, dos160 = aal.to(device), cc200.to(
                    device), dos160.to(device)
                aal_masked, cc200_masked, dos160_masked = aal_masked.to(
                    device), cc200_masked.to(device), dos160_masked.to(device)
                aal_mask, cc200_mask, dos160_mask = aal_mask.to(
                    device), cc200_mask.to(device), dos160_mask.to(device)
                optimizer.zero_grad()

                outputs = model(aal, cc200, dos160, aal_mask,
                                cc200_mask, dos160_mask)
                loss_aal = crit_aal(outputs[0], aal, aal_masked)
                loss_cc200 = crit_cc200(outputs[1], cc200, cc200_masked)
                loss_dos160 = crit_dos160(outputs[2], dos160, dos160_mask)
                loss = loss_aal + loss_cc200 + loss_dos160
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                epoch_losses.append(loss.item())
            if scheduler:
                scheduler.step()
            tepoch.set_postfix(loss=f"{running_loss/len(train_loader)}")
            losses.extend(epoch_losses)

            model.eval()
            with torch.no_grad():
                val_running_loss = 0.0
                for j, ((aal, cc200, dos160), (aal_masked, cc200_masked, dos160_masked), (aal_mask, cc200_mask, dos160_mask)) in enumerate(val_loader):
                    aal, cc200, dos160 = aal.to(device), cc200.to(
                        device), dos160.to(device)
                    aal_masked, cc200_masked, dos160_masked = aal_masked.to(
                        device), cc200_masked.to(device), dos160_masked.to(device)
                    aal_mask, cc200_mask, dos160_mask = aal_mask.to(
                        device), cc200_mask.to(device), dos160_mask.to(device)

                    outputs = model(aal, cc200, dos160, aal_mask,
                                    cc200_mask, dos160_mask)
                    loss_aal = crit_aal(outputs[0], aal, aal_masked)
                    loss_cc200 = crit_cc200(outputs[1], cc200, cc200_masked)
                    loss_dos160 = crit_dos160(outputs[2], dos160, dos160_mask)
                    loss = loss_aal + loss_cc200 + loss_dos160

                    val_running_loss += loss.item()
                    epoch_losses.append(loss.item())
                    val_losses.append(loss.item())

                avg_val_loss = val_running_loss / len(val_loader)

                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    counter = 0
                    best_model = copy.deepcopy(model)
                else:
                    counter += 1
                    if early_stopping and counter >= patience:
                        print("Early stopping!")
                        break
    return best_model


def train(model, train_loader, val_loader, criterion, optimizer, epochs, device, patience=2, scheduler=None, return_best_acc=False):

    early_stopping = True if patience else False
    losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_acc = 0
    counter = 0

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    with tqdm(range(epochs), unit="epoch") as tepoch:
        for i in tepoch:
            tepoch.set_description(f"Epoch {i}")
            epoch_losses = []
            running_loss = 0.0
            model.train()
            for i, (inputs, labels) in enumerate(train_loader):
                aal, cc200, do160 = inputs[0].to(
                    device), inputs[1].to(device), inputs[2].to(device)
                labels = labels.to(device)
                optimizer.zero_grad()

                outputs = model(aal, cc200, do160)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                epoch_losses.append(loss.item())
            if scheduler:
                scheduler.step()
            tepoch.set_postfix(loss=f"{running_loss/len(train_loader)}")
            losses.extend(epoch_losses)

            model.eval()
            with torch.no_grad():
                val_running_loss = 0.0
                y_true, y_pred = [], []
                for inputs, labels in val_loader:
                    aal, cc200, do160 = inputs[0].to(
                        device), inputs[1].to(device), inputs[2].to(device)
                    labels = labels.to(device)

                    outputs = model(aal, cc200, do160)
                    val_loss = criterion(outputs, labels)
                    val_running_loss += val_loss.item()

                    val_losses.append(val_loss.item())
                    y_true.extend(
                        np.argmax(labels.detach().cpu().numpy(), axis=1))

                    yp = outputs.detach().cpu().numpy()
                    y_pred.extend(np.argmax(np.where(yp > 0.5, 1, 0), axis=1))

                avg_val_loss = val_running_loss / len(val_loader)

                acc = accuracy_score(y_true, y_pred)

                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    best_model = copy.deepcopy(model)
                    counter = 0
                else:
                    counter += 1
                    if early_stopping and counter >= patience:
                        print("Early stopping!")
                        break
                if acc > best_acc:
                    best_acc = acc

    print(f"best acc: {best_acc}")

    return best_model, best_acc if return_best_acc else best_model


def test(model, test_loader, device):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for inputs, targets in test_loader:
            aal, cc200, do160 = inputs[0].to(
                device), inputs[1].to(device), inputs[2].to(device)
            targets = targets.to(device)
            outputs = model(aal, cc200, do160)
            y_true.extend(np.argmax(targets.detach().cpu().numpy(), axis=1))
            yp = F.softmax(outputs, dim=1).detach().cpu().numpy()
            y_pred.extend(np.argmax(np.where(yp > 0.5, 1, 0), axis=1))
    return y_true, y_pred
