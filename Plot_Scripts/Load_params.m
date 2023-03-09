%% Script to load learnt parameters and compute bounds. Used in some versions of the result_analysis scripts

%% Load learnt parameters

fname = [folder num2str(iter) '/weights_and_params.mat'];
load(fname, 'weights_and_params');
x_eq = weights_and_params('x_eq');
x1bound = weights_and_params('x1bound');
x2bound = weights_and_params('x2bound');
AG = weights_and_params('AG');
BG = weights_and_params('BG');
nG = size(AG, 1);
eta_NN = weights_and_params('eta_NN');
eta_ROA = weights_and_params('eta_ROA');
rho = weights_and_params('rho');

W = containers.Map;
n = containers.Map;
b = containers.Map;
for j=1:length(weights_and_params)-9 % 9 parameters stored
    key_W = sprintf('W%u',j);
    key_n = sprintf('n%u',j);
    key_b = sprintf('b%u',j);
    W(key_W) = weights_and_params(key_W);
    n(key_n) = size(W(key_W),1);
    b(key_b) = zeros(n(key_n),1);
end

nphi = 0; % number of hidden neurons
for j= 1:length(n)-1 % excludes output layer
    key = sprintf('n%u',j);
    nphi = nphi + n(key);
end

fname2 = [folder num2str(iter) '/sdpvar.mat'];
load(fname2, 'dec_var')
Q1 = dec_var.Q1;
Q2 = dec_var.Q2;
L1 = dec_var.L1;
L2 = dec_var.L2;
L3 = dec_var.L3;
L4 = dec_var.L4;
Yk = dec_var.Yk;

%% bounds for the inputs and outputs of the nonlinearities

% equilibrium points of the CL system - from (5) and (12) in Yin et al: stab
% analysis using QC for systems with NN controllers
% when bias terms are zero, this implies all eq. points in the NN are zero

v_eq = containers.Map;
w_eq = containers.Map;
v_eq('v1') = W('W1')*x_eq + b('b1');
w_eq('w1') = tanh(v_eq('v1'));

for j=2:length(n)
    key_v = sprintf('v%u',j);
    key_W = sprintf('W%u',j);
    key_w = sprintf('w%u',j-1);
    key_b = sprintf('b%u',j); 
    v_eq(key_v) = W(key_W)*w_eq(key_w) + b(key_b);
    if j < length(n)
        key_w = sprintf('w%u',j);
        w_eq(key_w) = tanh(v_eq(key_v));
    end
end

% -x1bound <= x1 <= x1bound; -x2bound <= x2 <= x2bound
x_ub = [x1bound; x2bound]; x_lb = [-x1bound; -x2bound];

% (6) from Gowal et al: Effectiveness of interval bound propagation ...
w_ub = containers.Map;
w_lb = containers.Map;
v_ub = containers.Map;
v_lb = containers.Map;

mu = 0.5*(x_ub + x_lb);
r = 0.5*(x_ub - x_lb);
mu = W('W1')*mu + b('b1');
r = abs(W('W1'))*r;
v_lb('v1') = mu - r;
v_ub('v1') = mu + r;
w_lb('w1') = tanh(v_lb('v1'));
w_ub('w1') = tanh(v_ub('v1'));

for j=2:length(n)
    key_w = sprintf('w%u',j-1);
    key_W = sprintf('W%u',j);
    key_b = sprintf('b%u',j);
    key_v = sprintf('v%u',j);

    mu = 0.5*(w_ub(key_w) + w_lb(key_w));
    r = 0.5*(w_ub(key_w) - w_lb(key_w));
    mu = W(key_W)*mu + b(key_b);
    r = abs(W(key_W))*r;
    v_lb(key_v) = mu - r;
    v_ub(key_v) = mu + r;
    if j < length(n) % no activation on the output layer
        key_w = sprintf('w%u',j);
        w_lb(key_w) = tanh(v_lb(key_v));
        w_ub(key_w) = tanh(v_ub(key_v));
    end
end

alpha = containers.Map;
for j=1:length(n)-1
    key_w = sprintf('w%u',j);
    key_v = sprintf('v%u',j);
    key_a = sprintf('a%u',j);
    a = min((w_ub(key_w)-w_eq(key_w))./(v_ub(key_v)-v_eq(key_v)), ...
        (w_eq(key_w)-w_lb(key_w))./(v_eq(key_v)-v_lb(key_v)));
    alpha(key_a) = a;
end

Alpha = blkdiag(diag(alpha('a1')));
for j=2:length(n)-1
    key_a = sprintf('a%u',j);
    Alpha = blkdiag(Alpha,diag(alpha(key_a)));
end

beta = 1.0;
Beta = beta*eye(nphi);

%% f(N)
N = blkdiag(W('W1'));
for j=2:length(n)
    key_W = sprintf('W%u',j);
    N = blkdiag(N, W(key_W));
end

% N = [Nvx, Nvw; Nux, Nuw]
Nvx = N(1:nphi,1:size(x_ub,1));
Nvw = N(1:nphi,size(x_ub,1)+1:end);
Nux = N(nphi+1:end,1:size(x_ub,1));
Nuw = N(nphi+1:end,size(x_ub,1)+1:end);

fNvx = inv(eye(nphi) - Nvw*1/2*(Alpha+Beta))*Nvx;
fNvw = inv(eye(nphi) - Nvw*1/2*(Alpha+Beta))*Nvw*1/2*(Beta-Alpha);
fNux = Nux + Nuw*1/2*(Alpha+Beta)*inv(eye(nphi)-Nvw*1/2*(Alpha+Beta))*Nvx;
fNuw = Nuw*1/2*(Beta-Alpha)+...
    Nuw*1/2*(Alpha+Beta)*inv(eye(nphi)-Nvw*1/2*(Alpha+Beta))*Nvw*1/2*(Beta-Alpha);

fN = [fNux, fNuw; fNvx, fNvw];

%% Plot and decision variables
Q =  blkdiag(Q1,Q2);
L = [L1, L2;...
     L3, L4];
