%% make_heston_data.m

nSamples = 10000;
d = 9; %[v0, theta, rho, kappa, xi, r, tau, k, S]

%Bounds for each parameter for LHS
%may need to fix S,K add d as a var
v0_l = 0.01; v0_h = 1.0;%initial variance (V0)
theta_l = 0.01; theta_h = 1.0;%long-run variance (ThetaV)
rho_l = -0.95; rho_h = 0.1; %correlation (RhoSV)
kappa_l = 0.1; kappa_h = 5.0;%mean reversion (Kappa)
xi_l = 0.01; xi_h = 1.0; %vol of vol (SigmaV)

r_l = -0.01; r_h = 0.1; %risk-free rate (Rate)
tau_l= 0.01; tau_h = 1.0; %maturity in years
k_l = -1.0; k_h = 1.0; %log-moneyness k = log(S/K)

S_l = 10; S_h = 6000; %spot (asset price) range

% lower/upper bounds vectors (1 x d)
l_bounds = [v0_l, theta_l, rho_l, kappa_l, xi_l, r_l, tau_l, k_l, S_l];
u_bounds = [v0_h, theta_h, rho_h, kappa_h, xi_h, r_h, tau_h, k_h, S_h];

%(LHS )Latin Hypercube sampling in (0,1)
U = lhsdesign(nSamples, d);   % from stat & ml toolbox

%Scale to actual ranges
X = l_bounds + U .* (u_bounds - l_bounds);   %nSamples x 9

% Unpack for readability
v0 = X(:,1);
theta = X(:,2);
rho = X(:,3);
kappa = X(:,4);
xi = X(:,5);    
r = X(:,6);
tau = X(:,7);
k = X(:,8);    
S = X(:,9);   

%% Price with built-in optByHestonNI (numerical integration)

OptSpec = "call"; %call options
Settle = datetime(2025,1,1); %arbitrary settle date
DividendYield = 0; %assume no dividends

prices = zeros(nSamples,1);
%get heston price with finance toolbox function
for i = 1:nSamples
    S_i = S(i);  
    K_i = S_i * exp(-k(i));  
    Rate_i = r(i);
    V0_i = v0(i);
    ThetaV_i = theta(i);
    Kappa_i = kappa(i);
    SigmaV_i = xi(i);
    RhoSV_i = rho(i);
    Maturity_i = Settle + years(tau(i)); %tau (years) -> datetime

    prices(i) = optByHestonNI( ...
        Rate_i, S_i, Settle, Maturity_i, OptSpec, K_i, ...
        V0_i, ThetaV_i, Kappa_i, SigmaV_i, RhoSV_i, ...
        'DividendYield', DividendYield); %numerical integration
end

%% Save dataset for Python (NN, loss, etc.)

%columns: [v0, theta, rho, kappa, xi, r, tau, k, S, price]
data = [X, prices];

%create table with variable names
varNames = {'v0', 'theta', 'rho', 'kappa', 'xi', 'r', 'tau', 'k', 'S', 'price'};
T = array2table(data, 'VariableNames', varNames);

outDir  = 'C:\Users\Owner\OneDrive\Desktop\projects\hestonmodel_NN';
outFile = fullfile(outDir, 'heston_dataset.csv');

writetable(T, outFile);   % writes a header row + data
