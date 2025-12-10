# market_surface.py
import torch
import pandas as pd

from neuralnettraining.bounds import l_bounds, range_bounds
from data.bs import bs_call_price_torch

class Market_Surface:
    def __init__(self, csv_path: str):
        df = pd.read_csv(csv_path)
        #rename to consistent names
        df = df.rename(columns={"S0": "S", "d": "q"})
        #tensors
        self.S   = torch.tensor(df["S"].values,   dtype=torch.float32).unsqueeze(1)   # (M,1)
        self.K   = torch.tensor(df["K"].values,   dtype=torch.float32).unsqueeze(1)   # (M,1)
        self.T_d = torch.tensor(df["T"].values,   dtype=torch.float32).unsqueeze(1)   # days
        self.r   = torch.tensor(df["r"].values,   dtype=torch.float32).unsqueeze(1)   # (M,1)
        self.q   = torch.tensor(df["q"].values,   dtype=torch.float32).unsqueeze(1)   # (M,1)
        self.iv  = torch.tensor(df["mktIV"].values, dtype=torch.float32).unsqueeze(1) # (M,1)
        self.M = self.S.shape[0]
        #tau in years
        self.tau = self.T_d / 365.0   # (M,1)
        #S/K (moneyness ratio)
        self.S_over_K = self.S / self.K   # (M,1)
        #Convert IV -> BS call price (all treated as calls)
        S_flat   = self.S.squeeze(1)      # (M,)
        K_flat   = self.K.squeeze(1)
        tau_flat = self.tau.squeeze(1)
        r_flat   = self.r.squeeze(1)
        q_flat   = self.q.squeeze(1)
        iv_flat  = self.iv.squeeze(1)
        self.price_mkt = bs_call_price_torch(
            S_flat, K_flat, r_flat, tau_flat, iv_flat, q_flat
        )  # (M,)
    def build_X_raw(self, theta_vec: torch.Tensor) -> torch.Tensor:
        if theta_vec.ndim == 2 and theta_vec.shape[1] == 1:
            theta_vec = theta_vec.squeeze(1)
        v0, theta, kappa, xi, rho = theta_vec  # each scalar tensor
        #repeat parameters across all M options
        v0_col    = v0.expand(self.M, 1)      # (M,1)
        theta_col = theta.expand(self.M, 1)
        kappa_col = kappa.expand(self.M, 1)
        xi_col    = xi.expand(self.M, 1)
        rho_col   = rho.expand(self.M, 1)
        X_raw = torch.cat([
            self.S_over_K, #(M,1)
            self.tau, #(M,1)
            self.r, #(M,1)
            self.q, #(M,1)
            v0_col, #(M,1)
            theta_col, #(M,1)
            kappa_col, #(M,1)
            xi_col, #(M,1)
            rho_col, #(M,1)
        ], dim=1) #(M,9)
        return X_raw
    def build_X_norm(self, theta_vec: torch.Tensor) -> torch.Tensor:
        X_raw = self.build_X_raw(theta_vec)         #(M,9)
        #l_bounds and range_bounds are (9,) so this broadcasts correctly
        X_norm = (X_raw - l_bounds) / range_bounds
        return X_norm
