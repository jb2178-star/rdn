# experiments/exp_architecture_ml.py
import time
import csv
from pathlib import Path

import torch
from torch import nn

from neuralnettraining.data_loading import make_dataloaders
from neuralnettraining.train_test import train_loop, test_loop

data_csv = "data/synthetic_datasets/heston_dataset_10000.csv"
results_csv = "results/architecture_comparison.csv"

batch_size = 64
train_frac = 0.7   
epochs = 50
weight_decay = 1e-5
lr_adam = 1e-3

hidden_layers_list = [2, 3, 4, 5]
hidden_dim_list = [32, 64, 128, 256]

Path("results").mkdir(exist_ok=True)

class simple_mlp(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_hidden, out_dim=1):
        super().__init__()
        layers = []
        #first hidden layer
        layers.append(nn.Linear(in_dim, hidden_dim))
        layers.append(nn.ReLU())
        #remaining hidden layers, iterate through to create the number of hidden layers
        for _ in range(num_hidden - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        #output layer
        layers.append(nn.Linear(hidden_dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
    
#store only test mse per (layers, neurons)
test_mse_matrix = {}
fieldnames = ["hidden_layers", "hidden_dim", "test_mse", "time_s"]
with open(results_csv, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()

    for hidden_layers in hidden_layers_list:
        for hidden_dim in hidden_dim_list:
            print(f"\n=== hidden_layers={hidden_layers}, hidden_dim={hidden_dim} ===")
            train_loader, test_loader = make_dataloaders(
                data_csv, batch_size, train_frac
            )
            model = simple_mlp(
                in_dim=9,
                hidden_dim=hidden_dim,
                num_hidden=hidden_layers,
                out_dim=1,
            )
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
                print(
                    f"  epoch {epoch:3d}/{epochs} | "
                    f"train mse={train_loss:.4e} | test mse={test_loss:.4e}"
                )
            elapsed = time.time() - t0
            final_test_mse = float(test_loss)
            writer.writerow(
                {
                    "hidden_layers": hidden_layers,
                    "hidden_dim": hidden_dim,
                    "test_mse": final_test_mse,
                    "time_s": elapsed,
                }
            )
            test_mse_matrix[(hidden_layers, hidden_dim)] = final_test_mse
print(f"\nsaved results to {results_csv}\n")
#print test mse table 
print("test mse (rows = hidden_layers, cols = hidden_dim)")
header = "layers\\nodes"
for hidden_dim in hidden_dim_list:
    header += f"   {hidden_dim:>8d}"
print(header)
for hidden_layers in hidden_layers_list:
    row = f"{hidden_layers:>6d}"
    for hidden_dim in hidden_dim_list:
        val = test_mse_matrix[(hidden_layers, hidden_dim)]
        row += f"   {val:8.4e}"
    print(row)
