import torch
import numpy as np
import pandas as pd

from models import MLP
from dataset import normalize_features
from predict import generate_submission

test_csv = "test.csv"
train_csv = "train.csv"
model_path = "best_model.pth"
output_csv = "submission.csv"

# Load data
train_df = pd.read_csv(train_csv)
test_df = pd.read_csv(test_csv)

feature_cols = [c for c in train_df.columns if c not in ["id", "label"]]

X_train = train_df[feature_cols].values.astype(np.float32)
X_test = test_df[feature_cols].values.astype(np.float32)
test_ids = test_df["id"].values

# Normalize using training stats
X_train, X_test = normalize_features(X_train, X_test)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
num_features = X_train.shape[1]
num_classes = len(np.unique(train_df["label"]))
model = MLP(num_features, num_classes).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))

# Generate submission
generate_submission(model, X_test, test_ids, output_csv, device)
