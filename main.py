# main.py

import time
import csv
from pathlib import Path

import torch
from torch import nn

from neuralnettraining.neural_net import NeuralNetModel1
from neuralnettraining.data_loading import make_dataloaders
from neuralnettraining.train_test import (
    train_loop,
    test_loop,
    price_error_stats,
    rel_price_error_stats,
)

def main():
    #reproducibility
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    csv_path = "data/synthetic_datasets/heston_dataset_50000.csv"
    #Hyper parematers
    batch_size = 64
    learning_rate = 1e-3
    epochs = 50
    train_frac = 0.7
    weight_decay = 1e-5
    #where to save stuff
    Path("results").mkdir(exist_ok=True)
    Path("models_").mkdir(exist_ok=True)
    results_csv = "results/main_training_50000.csv"
    fieldnames  = ["epoch", "train_mse", "test_mse", "time_s"]
    #data 
    train_loader, test_loader = make_dataloaders(csv_path, batch_size, train_frac)
    #model / loss / optimizer 
    model = NeuralNetModel1()
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    #training loop 
    t0 = time.time()
    with open(results_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for epoch in range(1, epochs + 1):
            train_loss = train_loop(
                train_loader,
                model,
                loss_fn,
                optimizer,
                verbose=False,
            )
            test_loss = test_loop(
                test_loader,
                model,
                loss_fn,
                verbose=False,
            )
            elapsed = time.time() - t0
            # print every 10 epochs (and first / last)
            if epoch == 1 or epoch == epochs or epoch % 10 == 0:
                print(
                    f"Epoch {epoch:3d}/{epochs}: "
                    f"train_loss={train_loss:.4e}, "
                    f"test_loss={test_loss:.4e}, "
                    f"time={elapsed:.1f}s"
                )
            writer.writerow(
                {
                    "epoch": epoch,
                    "train_mse": float(train_loss),
                    "test_mse": float(test_loss),
                    "time_s": elapsed,
                }
            )
    print("\nTraining finished.")
    print("\nPrice error stats on test set (in $):")
    price_error_stats(model, test_loader)
    print("\nRelative price error stats:")
    rel_price_error_stats(model, test_loader)
    #save trained model for calibration
    out_path = "models_/heston_nn_50000.pth"
    torch.save(model.state_dict(), out_path)
    print(f"\nSaved trained model to {out_path}")
if __name__ == "__main__":
    main()
