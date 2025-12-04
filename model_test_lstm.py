# model_test_lstm.py

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from load_data import load_data

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
THRESHOLD = 0.5

FEATURE_COLS = [...]  # same list


class SlipLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=32):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x: (batch, seq_len, 1)
        out, (h_n, c_n) = self.lstm(x)
        last = h_n.squeeze(0)
        logits = self.fc(last).squeeze(-1)
        return logits


def _load_xy():
    df_tr = load_data("bts3v2_palm_down")
    df_te = load_data("bts3v2_palm_down_test")

    X_train = df_tr[FEATURE_COLS].values.astype(np.float32)
    y_train = df_tr["slipped"].values.astype(np.float32)
    X_test = df_te[FEATURE_COLS].values.astype(np.float32)
    y_test = df_te["slipped"].values.astype(np.float32)

    mean = X_train.mean(axis=0, keepdims=True)
    std = X_train.std(axis=0, keepdims=True) + 1e-8
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    return X_train, y_train, X_test, y_test


def run(model_dir: str):
    os.makedirs(model_dir, exist_ok=True)

    X_train, y_train, X_test, y_test = _load_xy()

    X_train_t = torch.from_numpy(X_train).unsqueeze(-1).to(DEVICE)
    y_train_t = torch.from_numpy(y_train).to(DEVICE)
    X_test_t = torch.from_numpy(X_test).unsqueeze(-1).to(DEVICE)

    train_ds = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)

    model = SlipLSTM(input_size=1, hidden_size=32).to(DEVICE)
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
        # print(f"[LSTM] Epoch {epoch+1}/{n_epochs}, loss={epoch_loss:.4f}")

    model.eval()
    with torch.no_grad():
        logits_test = model(X_test_t).cpu().numpy()
        probs_test = 1.0 / (1.0 + np.exp(-logits_test))

    preds_test = (probs_test >= THRESHOLD).astype(int)

    acc = accuracy_score(y_test, preds_test)
    prec = precision_score(y_test, preds_test, zero_division=0)
    rec = recall_score(y_test, preds_test)
    f1 = f1_score(y_test, preds_test)

    model_path = os.path.join(model_dir, "slip_lstm.pt")
    torch.save(model.state_dict(), model_path)

    metrics = {
        "model": "lstm",
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "threshold": THRESHOLD,
        "model_path": model_path,
    }
    print("[LSTM] metrics:", metrics)
    return metrics


if __name__ == "__main__":
    run("model_test_models")
