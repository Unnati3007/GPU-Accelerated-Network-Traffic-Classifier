import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler, LabelEncoder
from pathlib import Path


LABEL_MAP = {
    "BENIGN": 0,
    "DDoS": 1,
    "PortScan": 2,
    "FTP-Patator": 3,
    "SSH-Patator": 3,   # merge brute-force variants
    "Web Attack": 4,
}

FEATURES_TO_DROP = [" Label", "Flow ID", " Source IP", " Destination IP",
                    " Timestamp", " Source Port"]


class CICIDSDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def load_and_preprocess(data_dir: str = "data/raw") -> tuple:
    dfs = []
    for p in Path(data_dir).glob("*.csv"):
        df = pd.read_csv(p, low_memory=False)
        df.columns = df.columns.str.strip()
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)

    # Map labels
    df["Label"] = df["Label"].map(
        lambda x: next((v for k, v in LABEL_MAP.items() if k in str(x)), -1)
    )
    df = df[df["Label"] != -1]

    # Drop non-feature columns
    drop_cols = [c for c in FEATURES_TO_DROP if c.strip() in df.columns]
    X = df.drop(columns=drop_cols + ["Label"]).select_dtypes(include=[np.number])
    y = df["Label"].values

    # Clean
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0).values.astype(np.float32)

    # Normalise
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return X, y, scaler
