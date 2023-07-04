import copy
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import torch.nn.functional as F


def train_(model, train_loader, val_loader, criterion, optimizer, epochs, device, patience=2, scheduler=None, return_best_acc=False):

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

                    # outputs = model(inputs, mask)
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

