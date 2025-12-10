# experiments/exp_optimizer_ml.py
# experiments/exp_optimizer_ml.py

import time
import csv
from pathlib import Path
import torch
from torch import nn
from neuralnettraining.data_loading import make_dataloaders
from neuralnettraining.neural_net_test import NeuralNetModel1
from neuralnettraining.train_test import train_loop 

data_csv = "data/synthetic_datasets/heston_dataset_10000.csv"
results_csv = "results/optimizer_comparison.csv"

batch_size = 64
train_frac = 0.7
epochs = 50
weight_decay = 1e-5
lr_sgd = 1e-3
lr_adam = 1e-3
Path("results").mkdir(exist_ok=True)
fieldnames = ["optimizer", "epoch", "train_mse", "time_s"]
final_stats = {}
with open(results_csv, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for opt_name in ["sgd", "adam"]:
        train_loader, _ = make_dataloaders(data_csv, batch_size, train_frac)
        model = NeuralNetModel1()
        loss_fn = nn.MSELoss()
        if opt_name == "sgd":
            optimizer = torch.optim.SGD(
                model.parameters(),
                lr=lr_sgd,
                weight_decay=weight_decay,
            )
        if opt_name == "adam":
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=lr_adam,
                weight_decay=weight_decay,
            )
        t0 = time.time()
        for epoch in range(1, epochs + 1):
            train_loss = train_loop(train_loader, model, loss_fn, optimizer, verbose=False)
            elapsed    = time.time() - t0
            print(
                f"[{opt_name}] epoch {epoch:3d}/{epochs} | "
                f"train mse={train_loss:.4e} | time={elapsed:.1f}s"
            )
            writer.writerow({
                "optimizer": opt_name,
                "epoch": epoch,
                "train_mse": float(train_loss),
                "time_s": elapsed,
            })
        final_stats[opt_name] = {
            "final_train_mse": float(train_loss),
            "total_time_s": elapsed,
        }
print(f"\nsaved results to {results_csv}\n")
for opt_name in ["sgd", "adam"]:
    stats = final_stats[opt_name]
    print(
        f"{opt_name.upper()} summary:"
        f"\n  final train mse: {stats['final_train_mse']:.4e}"
        f"\n  total time    : {stats['total_time_s']:.2f} s\n"
    )
