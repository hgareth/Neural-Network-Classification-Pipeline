import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, random_split

from models import MLP
from dataset import FeaturesDataset, normalize_features
from train import train_one_epoch, evaluate, LabelSmoothingCrossEntropy

# ------------------------
# Settings
# ------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

train_csv = "train.csv"   # or "data/train.csv" if you use a data/ folder

# ------------------------
# Load CSV
# ------------------------
df = pd.read_csv(train_csv)

feature_cols = [c for c in df.columns if c not in ["id", "label"]]

X = df[feature_cols].values.astype(np.float32)
y = df["label"].values.astype(int)

num_features = X.shape[1]
num_classes = len(np.unique(y))
print(f"Num features: {num_features}, Num classes: {num_classes}")

# ------------------------
# Normalize features
# ------------------------
X, _ = normalize_features(X, X)

# ------------------------
# Dataset + split
# ------------------------
dataset = FeaturesDataset(X, y)

val_ratio = 0.1
val_size = int(len(dataset) * val_ratio)
train_size = len(dataset) - val_size

train_dataset, val_dataset = random_split(
    dataset,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)

print(f"Train samples: {train_size}, Val samples: {val_size}")

batch_size = 256   # smaller batch for potentially better generalization

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# ------------------------
# Model, loss, optimizer, scheduler
# ------------------------
model = MLP(num_features, num_classes).to(DEVICE)

# Label smoothing loss
criterion = LabelSmoothingCrossEntropy(smoothing=0.1)

# Slightly higher LR + weight decay
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=2e-3,
    weight_decay=1e-4
)

# LR scheduler: every 7 epochs, multiply LR by 0.5
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=7,
    gamma=0.5
)

# ------------------------
# Training loop
# ------------------------
epochs = 40
best_val_acc = 0.0
best_state = None

for epoch in range(1, epochs + 1):
    train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
    val_loss, val_acc = evaluate(model, val_loader, criterion, DEVICE)

    print(
        f"Epoch {epoch}/{epochs} | "
        f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
        f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
    )

    scheduler.step()

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_state = model.state_dict()

# ------------------------
# Save best model
# ------------------------
if best_state is not None:
    torch.save(best_state, "best_model.pth")

print("Training finished. Best Val Acc:", best_val_acc)
