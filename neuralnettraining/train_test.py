# train_test.py
import torch

def train_loop(dataloader, model, loss_fn, optimizer, verbose=False):
    model.train()
    total_loss = 0.0
    num_batches = 0

    for i, (X, y) in enumerate(dataloader):
        pred = model(X)
        loss = loss_fn(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        num_batches += 1

        if verbose and i % 100 == 0:
            loss_val = loss.item()
            current_step = i * X.size(0)
            print(
                f"batch {i:4d}, "
                f"loss: {loss_val:>10.6f}  "
                f"[{current_step:>5d}/{len(dataloader.dataset):>5d}]"
            )

    avg_loss = total_loss / max(num_batches, 1)
    return avg_loss


@torch.no_grad()
def test_loop(dataloader, model, loss_fn, verbose=True):
    model.eval()
    total_loss = 0.0
    num_batches = 0

    for X, y in dataloader:
        pred = model(X)
        loss = loss_fn(pred, y)

        total_loss += loss.item()
        num_batches += 1

    avg_loss = total_loss / max(num_batches, 1)
    rmse = avg_loss ** 0.5

    if verbose:
        print(f"Test Error: Avg MSE: {avg_loss:>10.6f}, RMSE: {rmse:>10.6f}")

    return avg_loss


@torch.no_grad()
def price_error_stats(model, dataloader, S_fixed: float = 100.0):
    """
    For synthetic data: S is constant = S_fixed (100 by construction).
    y_rel is price/S, model(X) predicts price/S as well.
    """
    model.eval()
    abs_errs = []

    for X_norm, y_rel in dataloader:
        pred_rel = model(X_norm)          #NN outputs price/S
        price_true = y_rel * S_fixed      #true price
        price_pred = pred_rel * S_fixed   #predicted price

        abs_err = (price_pred - price_true).abs()
        abs_errs.append(abs_err)

    abs_errs = torch.cat(abs_errs)

    print("Mean abs price error ($):   ", abs_errs.mean().item())
    print("Median abs price error ($): ", abs_errs.median().item())


@torch.no_grad()
def rel_price_error_stats(model, dataloader):
    """
    Relative error in *relative price* space; since S is constant, this is
    also the relative error in actual prices.
    """
    model.eval()
    abs_rel_errs = []

    for X_norm, y_rel in dataloader:
        pred_rel = model(X_norm)
        #relative error in C/S
        rel_err = (pred_rel - y_rel).abs() / (y_rel.abs() + 1e-6)
        abs_rel_errs.append(rel_err)

    abs_rel_errs = torch.cat(abs_rel_errs)
    print("Mean relative price error:", abs_rel_errs.mean().item())
    print("Median relative price error:", abs_rel_errs.median().item())
