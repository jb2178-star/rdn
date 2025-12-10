# experiments/exp_opt_arch_datasets.py

import time
import csv
from pathlib import Path
import torch
from torch import nn
from neuralnettraining.data_loading import make_dataloaders
from neuralnettraining.neural_net import NeuralNetModel1
from neuralnettraining.train_test import train_loop, test_loop
#dataset sizes and corresponding file pattern
dataset_sizes = [1000, 10000, 30000, 50000]
data_pattern = "data/synthetic_datasets/heston_dataset_{}.csv"
results_csv = "results/optimal_arch_datasets.csv"
batch_size   = 64
train_frac   = 0.7
epochs       = 50
weight_decay = 1e-5
lr_adam      = 1e-3

Path("results").mkdir(exist_ok=True)
fieldnames = ["dataset_size", "epoch", "train_mse", "test_mse", "time_s"]
final_stats = {}
with open(results_csv, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for n in dataset_sizes:
        csv_path = data_pattern.format(n)
        print(f"\n=== dataset_size={n}, csv='{csv_path}' ===")
        train_loader, test_loader = make_dataloaders(csv_path, batch_size, train_frac)
        #instantiate the model (instance, not class)
        model = NeuralNetModel1()
        loss_fn = nn.MSELoss()
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr_adam,
            weight_decay=weight_decay,
        )
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
                f"[N={n}] epoch {epoch:3d}/{epochs} | "
                f"train mse={train_loss:.4e} | test mse={test_loss:.4e} | "
                f"time={elapsed:.1f}s"
            )
            writer.writerow(
                {
                    "dataset_size": n,
                    "epoch": epoch,
                    "train_mse": float(train_loss),
                    "test_mse": float(test_loss),
                    "time_s": elapsed,
                }
            )
        final_stats[n] = {
            "final_train_mse": float(train_loss),
            "final_test_mse": float(test_loss),
            "total_time_s": elapsed,
        }
print(f"\nsaved results to {results_csv}\n")
for n in dataset_sizes:
    stats = final_stats[n]
    print(
        f"N={n} summary:"
        f"\n  final train mse: {stats['final_train_mse']:.4e}"
        f"\n  final test  mse: {stats['final_test_mse']:.4e}"
        f"\n  total time    : {stats['total_time_s']:.2f} s\n"
    )
