# calibration.py (or wherever you keep calibration utilities)
import torch

def project_theta(theta: torch.Tensor) -> None:
    with torch.no_grad():
        # --- box constraints, updated to new bounds ---
        theta[0].clamp_(0.001, 0.20)   # v0
        theta[1].clamp_(0.020, 0.20)   # theta (long-run variance)
        theta[2].clamp_(0.5,   8.0)    # kappa
        theta[3].clamp_(0.2,   3.0)    # xi (vol-of-vol)
        theta[4].clamp_(-0.99, 0.0)  # rho

        # Feller: 2*kappa*theta >= xi^2  (softly enforced via xi cap)
        v0     = theta[0]
        thetaL = theta[1]
        kappa  = theta[2]
        xi     = theta[3]

        xi_max = torch.sqrt(torch.clamp(2.0 * kappa * thetaL, min=1e-8))
        if xi > xi_max:
            theta[3].copy_(xi_max)

def nn_calibration_loss(theta: torch.Tensor, model, market_surface) -> torch.Tensor:
    #Build normalized inputs for current theta
    #shape: (M, 9)
    X_norm = market_surface.build_X_norm(theta)
    #NN outputs price_relative = price / S
    price_rel_pred = model(X_norm).squeeze(-1)   #(M,)
    S = market_surface.S.squeeze(1)              #(M,)
    price_pred = price_rel_pred * S              #(M,)
    price_mkt = market_surface.price_mkt         #(M,)
    loss = torch.mean((price_pred - price_mkt) ** 2)
    return loss
def nn_calibration_residual(theta: torch.Tensor, model, market_surface) -> torch.Tensor:
    X_norm = market_surface.build_X_norm(theta)        # (M, 9)

    price_rel_pred = model(X_norm).squeeze(-1)         # (M,)
    S = market_surface.S.squeeze(1)                    # (M,)
    price_pred = price_rel_pred * S                    # (M,)

    price_mkt = market_surface.price_mkt               # (M,)

    residual = price_pred - price_mkt                  # (M,)
    return residual



