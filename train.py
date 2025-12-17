import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score

class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing: float = 0.1):
        """
        Label smoothing: instead of a one-hot (1,0,0,...),
        use 1 - smoothing for the true class and
        smoothing/(num_classes-1) for others.
        """
        super().__init__()
        assert 0.0 <= smoothing < 1.0
        self.smoothing = smoothing

    def forward(self, logits, target):
        # logits: (batch_size, num_classes)
        # target: (batch_size,) int64
        num_classes = logits.size(1)
        log_probs = torch.log_softmax(logits, dim=1)

        with torch.no_grad():
            # Create smoothed target distribution
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.smoothing / (num_classes - 1))
            true_dist.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)

        loss = (-true_dist * log_probs).sum(dim=1).mean()
        return loss


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    preds_list = []
    targets_list = []

    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * X_batch.size(0)
        preds = outputs.argmax(dim=1).cpu().numpy()
        preds_list.append(preds)
        targets = y_batch.cpu().numpy()
        targets_list.append(targets)

    epoch_loss = running_loss / len(loader.dataset)
    preds_list = np.concatenate(preds_list)
    targets_list = np.concatenate(targets_list)
    epoch_acc = accuracy_score(targets_list, preds_list)

    return epoch_loss, epoch_acc


def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    preds_list = []
    targets_list = []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            running_loss += loss.item() * X_batch.size(0)

            preds = outputs.argmax(dim=1).cpu().numpy()
            preds_list.append(preds)
            targets_list.append(y_batch.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    preds_list = np.concatenate(preds_list)
    targets_list = np.concatenate(targets_list)
    epoch_acc = accuracy_score(targets_list, preds_list)

    return epoch_loss, epoch_acc
