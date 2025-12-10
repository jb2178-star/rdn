# data.py
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split
from neuralnettraining.bounds import l_bounds, range_bounds

class HestonDatasetFromCSV(Dataset):
    def __init__(self, csv_path: str):
        df = pd.read_csv(csv_path)
        #[S_over_K, tau, r, q, v0, theta, kappa, xi, rho]
        feature_cols = [
            "S_over_K",  # 0
            "tau",       # 1
            "r",         # 2
            "q",         # 3
            "v0",        # 4
            "theta",     # 5
            "kappa",     # 6
            "xi",        # 7
            "rho"        # 8
        ]
        target_col = "price"

        X_np = df[feature_cols].to_numpy()
        y_np = df[target_col].to_numpy()
        S_np = df["S"].to_numpy()  #spot used to compute relative price

        #eliminate unrealistic data, where price is negative
        y_np = y_np.copy()
        y_np[y_np < 0.0] = 0.0

        #(N, 9) float tensor
        X = torch.tensor(X_np, dtype=torch.float32)

        #normalize with same bounds as synthetic data generation
        #l_bounds, range_bounds are (9,) tensors
        X = (X - l_bounds) / range_bounds

        self.X = X

        #relative price = price / S   
        y_rel = y_np / S_np
        self.y = torch.tensor(y_rel, dtype=torch.float32).unsqueeze(1)  #(N,1)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def make_dataloaders(csv_path: str, batch_size: int, train_frac: float):
    full_dataset = HestonDatasetFromCSV(csv_path)

    n_total = len(full_dataset)
    n_train = int(train_frac * n_total)
    n_test = n_total - n_train

    train_dataset, test_dataset = random_split(full_dataset, [n_train, n_test])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)

    return train_loader, test_loader