# bounds.py
import torch

import torch

l_bounds = torch.tensor([
    0.5, #S_over_K_l
    0.015, #tau_l
   -0.02,  #r_l
    0.0, #q_l (dividend yield d)

    #Heston params 
    0.001, # v0_l
    0.020,  # theta_l
    0.5,    # kappa_l
    0.2,    # xi_l
   -0.99    # rho_l
], dtype=torch.float32)

u_bounds = torch.tensor([
    5.0,    # S_over_K_h
    3.0,    # tau_h
    0.10,   # r_h
    0.10,   # q_h

    #Heston params 
    0.20, #v0_h
    0.20, #theta_h
    8.0, #kappa_h
    3.0, #xi_h
    0.0 #rho_h
], dtype=torch.float32)

range_bounds = u_bounds - l_bounds
