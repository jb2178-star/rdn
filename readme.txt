Heston Calibration with Neural Networks
---------------------------------------
A pytorch-based neural network that approximates heston call prices, and calibrates to return heston parameters
It first generates Latin Hypercube Sampling for sythetic data in matlab with the heston price as the ground truth
Then it calibrates with option prices to generate the heston parameters.





for expirements 
run in cd python -m experiments.exp_optimizer_nn
python -m experiments.exp_architecture
python -m experiments.exp_low_vs_high_arch
python -m experiments.exp_total
python -m experiments.exp_calibration_lm