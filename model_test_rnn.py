# model_test_rnn.py

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from load_data import load_data

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
THRESHOLDS = [0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.15, 0.2, 0.25, 0.30, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]


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

class SlipRNN(nn.Module):
    def __init__(self, input_size=1, hidden_size=32):
        super().__init__()
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x: (batch, seq_len, 1)
        out, h_n = self.rnn(x)   # h_n: (1, batch, hidden)
        last = h_n.squeeze(0)    # (batch, hidden)
        logits = self.fc(last).squeeze(-1)
        return logits

def normalise_per_object(df, feature_cols):
    df_norm = df.copy()
    for obj_id, grp in df.groupby("object"):
        X = grp[feature_cols].values.astype(np.float32)
        mean = X.mean(axis=0, keepdims=True)
        std = X.std(axis=0, keepdims=True) + 1e-8
        Xn = (X - mean) / std
        df_norm.loc[grp.index, feature_cols] = Xn
    return df_norm

def _load_xy():
    df_tr = load_data("bts3v2_palm_down")
    df_te = load_data("bts3v2_palm_down_test")


    df_tr[FEATURE_COLS] = df_tr[FEATURE_COLS].astype(np.float32)
    df_te[FEATURE_COLS] = df_te[FEATURE_COLS].astype(np.float32)


    df_tr = normalise_per_object(df_tr, FEATURE_COLS)
    df_te = normalise_per_object(df_te, FEATURE_COLS)

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

    # reshape for RNN: (N, seq_len, 1)
    X_train_t = torch.from_numpy(X_train).unsqueeze(-1).to(DEVICE)
    y_train_t = torch.from_numpy(y_train).to(DEVICE)
    X_test_t = torch.from_numpy(X_test).unsqueeze(-1).to(DEVICE)

    train_ds = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)

    model = SlipRNN(input_size=1, hidden_size=32).to(DEVICE)
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
        # print(f"[RNN] Epoch {epoch+1}/{n_epochs}, loss={epoch_loss:.4f}")

    model.eval()
    with torch.no_grad():
        logits_test = model(X_test_t).cpu().numpy()
        probs_test = 1.0 / (1.0 + np.exp(-logits_test))

    for THRESHOLD in THRESHOLDS:
        preds_test = (probs_test >= THRESHOLD).astype(int)

        acc = accuracy_score(y_test, preds_test)
        prec = precision_score(y_test, preds_test, zero_division=0)
        rec = recall_score(y_test, preds_test)
        f1 = f1_score(y_test, preds_test)

        model_path = os.path.join(model_dir, "slip_rnn.pt")
        torch.save(model.state_dict(), model_path)

        metrics = {
            "model": "rnn",
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "threshold": THRESHOLD,
            "model_path": model_path,
        }
        print("[RNN] metrics:", metrics)
    return metrics


if __name__ == "__main__":
    run("model_test_models")
