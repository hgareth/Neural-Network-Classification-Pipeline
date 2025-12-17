import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from dataset import FeaturesDataset

def generate_submission(model, X_test, test_ids, outfile, device):
    model.eval()
    test_dataset = FeaturesDataset(X_test, y=None)
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)

    all_preds = []

    with torch.no_grad():
        for X_batch in test_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.append(preds)

    all_preds = np.concatenate(all_preds)

    submission = pd.DataFrame({
        "id": test_ids,
        "label": all_preds
    })

    submission.to_csv(outfile, index=False)
    print(f"Saved submission file: {outfile}")