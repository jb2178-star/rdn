import time
import csv
from pathlib import Path

import torch
from torch import nn

from neuralnettraining.data_loading import make_dataloaders
from neuralnettraining.train_test import train_loop, test_loop
data_csv = "data/synthetic_datasets/heston_dataset_10000.csv"
results_csv = "results/low_vs_high.csv"

batch_size = 64
train_frac = 0.7   
epochs = 50
weight_decay = 1e-5
lr_adam = 1e-3

Path("results").mkdir(exist_ok=True)

class neural_net_high(nn.Module):
    def __init__(self, in_dim=9, hidden_dim=32, out_dim=1):
        super().__init__()
        self.net = nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, out_dim)#outputs relative price C/S  
    )

    def forward(self, x):
        return self.net(x)
    
class neural_net_low(nn.Module):
    def __init__(self, in_dim=9, hidden_dim=256, out_dim=1):
        super().__init__()
        self.net = nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, out_dim)#outputs relative price C/S  
    )


    def forward(self, x):
        return self.net(x)
    
#shared dataloaders
train_loader, test_loader = make_dataloaders(data_csv, batch_size, train_frac)
fieldnames = ["arch", "epoch", "train_mse", "test_mse", "time_s"]
final_stats = {}
with open(results_csv, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for arch_name in ["low", "high"]:
        if arch_name == "low":
            model = neural_net_low()
        if arch_name == "high":
            model = neural_net_high()
        loss_fn = nn.MSELoss()
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr_adam,
            weight_decay=weight_decay,
        )
        print(f"\n=== training architecture: {arch_name} ===")
        t0 = time.time()
        for epoch in range(1, epochs + 1):
            train_loss = train_loop(
                train_loader, model, loss_fn, optimizer, verbose=False
            )
            test_loss = test_loop(
                test_loader, model, loss_fn, verbose=False
            )
            elapsed = time.time() - t0

            print(
                f"[{arch_name}] epoch {epoch:3d}/{epochs} | "
                f"train mse={train_loss:.4e} | test mse={test_loss:.4e} | "
                f"time={elapsed:.1f}s"
            )
            writer.writerow(
                {
                    "arch": arch_name,
                    "epoch": epoch,
                    "train_mse": float(train_loss),
                    "test_mse": float(test_loss),
                    "time_s": elapsed,
                }
            )
        # store final stats after last epoch
        final_stats[arch_name] = {
            "final_train_mse": float(train_loss),
            "final_test_mse": float(test_loss),
            "total_time_s": elapsed,
        }
print(f"\nsaved results to {results_csv}\n")
# print summary
for arch_name in ["low", "high"]:
    stats = final_stats[arch_name]
    print(
        f"{arch_name.upper()} summary:"
        f"\n  final train mse: {stats['final_train_mse']:.4e}"
        f"\n  final test  mse: {stats['final_test_mse']:.4e}"
        f"\n  total time    : {stats['total_time_s']:.2f} s\n"
    )