# bs.py
import torch

def bs_call_price_torch(S, K, r, tau, sigma, q):
    """
    Black–Scholes call price in torch (vectorized).

    S, K, r, tau, sigma, q: 1D tensors of same length.
    tau in years, sigma annualized vol.
    """
    eps = 1e-8
    sigma = torch.clamp(sigma, min=eps)
    tau   = torch.clamp(tau,   min=eps)

    sqrtT = torch.sqrt(tau)
    d1 = (torch.log(S / K) + (r - q + 0.5 * sigma**2) * tau) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT

    normal = torch.distributions.Normal(0.0, 1.0)
    Nd1 = normal.cdf(d1)
    Nd2 = normal.cdf(d2)

    disc_q = torch.exp(-q * tau)
    disc_r = torch.exp(-r * tau)

    call_price = S * disc_q * Nd1 - K * disc_r * Nd2
    return call_price
