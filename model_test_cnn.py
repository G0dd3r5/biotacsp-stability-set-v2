# model_test_cnn.py

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from load_data import load_data

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
THRESHOLD = 0.5  # change later if you want


FEATURE_COLS = [
    'ff_biotac_1','ff_biotac_2','ff_biotac_3','ff_biotac_4','ff_biotac_5',
    'ff_biotac_6','ff_biotac_7','ff_biotac_8','ff_biotac_9','ff_biotac_10',
    'ff_biotac_11','ff_biotac_12','ff_biotac_13','ff_biotac_14','ff_biotac_15',
    'ff_biotac_16','ff_biotac_17','ff_biotac_18','ff_biotac_19','ff_biotac_20',
    'ff_biotac_21','ff_biotac_22','ff_biotac_23','ff_biotac_24',
    'mf_biotac_1','mf_biotac_2','mf_biotac_3','mf_biotac_4','mf_biotac_5',
    'mf_biotac_6','mf_biotac_7','mf_biotac_8','mf_biotac_9','mf_biotac_10',
    'mf_biotac_11','mf_biotac_12','mf_biotac_13','mf_biotac_14','mf_biotac_15',
    'mf_biotac_16','mf_biotac_17','mf_biotac_18','mf_biotac_19','mf_biotac_20',
    'mf_biotac_21','mf_biotac_22','mf_biotac_23','mf_biotac_24',
    'th_biotac_1','th_biotac_2','th_biotac_3','th_biotac_4','th_biotac_5',
    'th_biotac_6','th_biotac_7','th_biotac_8','th_biotac_9','th_biotac_10',
    'th_biotac_11','th_biotac_12','th_biotac_13','th_biotac_14','th_biotac_15',
    'th_biotac_16','th_biotac_17','th_biotac_18','th_biotac_19','th_biotac_20',
    'th_biotac_21','th_biotac_22','th_biotac_23','th_biotac_24'
]


class SlipCNN(nn.Module):
    def __init__(self, seq_len: int = 72):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        # x: (batch, 1, 72)
        x = self.conv(x)
        x = self.pool(x).squeeze(-1)  # (batch, 64)
        logits = self.fc(x).squeeze(-1)  # (batch,)
        return logits


def _load_xy():
    df_tr = load_data("bts3v2_palm_down")
    df_te = load_data("bts3v2_palm_down_test")

    X_train = df_tr[FEATURE_COLS].values.astype(np.float32)
    y_train = df_tr["slipped"].values.astype(np.float32)
    X_test = df_te[FEATURE_COLS].values.astype(np.float32)
    y_test = df_te["slipped"].values.astype(np.float32)

    # simple standardisation based on train
    mean = X_train.mean(axis=0, keepdims=True)
    std = X_train.std(axis=0, keepdims=True) + 1e-8
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    return X_train, y_train, X_test, y_test


def run(model_dir: str):
    os.makedirs(model_dir, exist_ok=True)

    X_train, y_train, X_test, y_test = _load_xy()

    # reshape for CNN: (N, 1, 72)
    X_train_t = torch.from_numpy(X_train).unsqueeze(1).to(DEVICE)
    y_train_t = torch.from_numpy(y_train).to(DEVICE)
    X_test_t = torch.from_numpy(X_test).unsqueeze(1).to(DEVICE)

    train_ds = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)

    model = SlipCNN(seq_len=X_train.shape[1]).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)

    n_epochs = 30

    model.train()
    for epoch in range(n_epochs):
        epoch_loss = 0.0
        for xb, yb in train_loader:
            optimiser.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimiser.step()
            epoch_loss += loss.item() * xb.size(0)
        epoch_loss /= len(train_loader.dataset)
        # print(f"[CNN] Epoch {epoch+1}/{n_epochs}, loss={epoch_loss:.4f}")

    # evaluation
    model.eval()
    with torch.no_grad():
        logits_test = model(X_test_t).cpu().numpy()
        probs_test = 1.0 / (1.0 + np.exp(-logits_test))

    preds_test = (probs_test >= THRESHOLD).astype(int)

    acc = accuracy_score(y_test, preds_test)
    prec = precision_score(y_test, preds_test, zero_division=0)
    rec = recall_score(y_test, preds_test)
    f1 = f1_score(y_test, preds_test)

    # save model
    model_path = os.path.join(model_dir, "slip_cnn.pt")
    torch.save(model.state_dict(), model_path)

    metrics = {
        "model": "cnn",
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "threshold": THRESHOLD,
        "model_path": model_path,
    }
    print("[CNN] metrics:", metrics)
    return metrics


if __name__ == "__main__":
    run("model_test_models")
