% Region of attraction analysis on the pendulum system with NN controller
% Modified: 09/03/23 - CRR - Modified code to allow arbitrary choice of NN depth and width.
function dec_var = solve_sdp(param)

% convert struct to map and define system parameters
param = containers.Map(fieldnames(param), struct2cell(param));

QC_type = param('QC_type');
AG = param('AG');
BG = param('BG');
nG = size(AG, 1);
nu = size(BG, 2);

%% load weights and biases of the NN controller

key_arr = []; % list of keys in string format
key_list = keys(param); % list of keys in cell format
for j=1:length(key_list)
    key_arr = [key_arr; string(key_list(j))];
end

W = containers.Map; % weights map
n = containers.Map; % size of each layer map
b = containers.Map; % biases map    
for j = 1:length(key_arr)
    key = sprintf('W%u',j);
    for k=1:length(key_arr)
        if key == key_arr(k)
            W(key) = param(key);
            temp = size(W(key),1);
            key = sprintf('n%u',j);
            n(key) = temp;
            temp = n(key);
            key = sprintf('b%u',j);
            b(key) = zeros(temp,1);
        end
    end
end

nphi = 0; % number of hidden neurons
for j= 1:length(n)-1 % excludes output layer
    key = sprintf('n%u',j);
    nphi = nphi + n(key);
end

disp("MATLAB: LOADED WEIGHTS")
%% bounds for the inputs and outputs of the nonlinearities

% equilibrium points of the CL system - from (5) and (12) in Yin et al: stab
% analysis using QC for systems with NN controllers
% when bias terms are zero, this implies all eq. points in the NN are zero

x_eq = param('x_eq');

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
x1bound = param('x1bound');
x2bound = param('x2bound');
x_ub = [x1bound; x2bound];
x_lb = -x_ub;

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
disp("MATLAB: COMPUTED BOUNDS ON INPUT/OUTPUT OF ACTIVATION FUNCTIONS")

%% f(N)
N = blkdiag(W('W1'));
for j=2:length(n)
    key_W = sprintf('W%u',j);
    N = blkdiag(N, W(key_W));
end

% N = [Nvx, Nvw; Nux, Nuw]
Nvx = N(1:nphi,1:nG);
Nvw = N(1:nphi,nG+1:end);
Nux = N(nphi+1:end,1:nG);
Nuw = N(nphi+1:end,nG+1:end);

fNvx = inv(eye(nphi) - Nvw*1/2*(Alpha+Beta))*Nvx;
fNvw = inv(eye(nphi) - Nvw*1/2*(Alpha+Beta))*Nvw*1/2*(Beta-Alpha);
fNux = Nux + Nuw*1/2*(Alpha+Beta)*inv(eye(nphi)-Nvw*1/2*(Alpha+Beta))*Nvx;
fNuw = Nuw*1/2*(Beta-Alpha)+...
    Nuw*1/2*(Alpha+Beta)*inv(eye(nphi)-Nvw*1/2*(Alpha+Beta))*Nvw*1/2*(Beta-Alpha);

fN = [fNux, fNuw; fNvx, fNvw];
disp("MATLAB: COMPUTED f(N)")

%% Convex Optimization - compute ROA
cvx_begin sdp
    cvx_solver Mosek_2
    
    % Variables
    variable Q1(nG,nG) symmetric;
    variable L1(nu, nG); % L1 = \tilde{Nux}*Q1;
    variable L2(nu, nphi); % L2 = \tilde{Nuw}*Q2;
    variable L3(nphi, nG); % L3 = \tilde{Nvx}*Q1;
    variable L4(nphi, nphi); % L4 = \tilde{Nvw}*Q2;
    
    Q1 >= 1e-8*eye(nG);    
    
    % conditions on Q2 dependent on choice of QC
    if QC_type == "CSB"
        variable Q2(nphi,nphi) diagonal;
        Q2 >= 1e-8*eye(nphi);
    elseif QC_type == "RA"
        %% Constraints on Q2 - strictly ultrametric matrix (Varga)
        variable Q2(nphi,nphi) symmetric; % changed to implement the repeated nonlinearities constraint!
    
        % property i (symmetry of Q2 automatically imposed)
        for i=1:size(Q2,1)
            for j=i:size(Q2,1)
                Q2(i,j)>=0.001;
            end
        end
        disp("MATLAB: CALCULATED CONSTRAINT i ON Q2")
        
        % property iii
        for i=1:size(Q2,1)
            for j=1:size(Q2,1)
                if i~=j
                    Q2(i,i)>=1.001*Q2(i,j);
                end
            end
        end
        disp("MATLAB: CALCULATED CONSTRAINT iii ON Q2")
    
        % property ii - these constraints are really slow to compute
        for i=1:size(Q2,1) %  avoids diagonal elements and repeated constraints due to symmetry
            for j=i+1:size(Q2,1)
                for k=1:size(Q2,1)
                    if k~=i & k~=j
                        Q2(i,j)>=Q2(i,k);
                        Q2(i,j)>=Q2(j,k);
                    end
                end
            end
        end
    else
        msg = 'QC_type must be CSB or RA';
        error(msg);
    end

    disp("MATLAB: CALCULATED CONSTRAINTS ON Q2")
    %% Continuing optimisation - Remainder of code remains unchanged!

    L = [L1, L2;...
     L3, L4];
    Q =  blkdiag(Q1,Q2);

    % left upper corner
    LU = Q;
    
    % right lower corner
    RL = Q;
    
    % left lower corber
    LL = [AG*Q1+BG*L1, BG*L2;...
          L3, L4];
    
    % right upper corner
    RU = LL';
    
    % Matrix Inequality
    LMI = [LU, RU;...
           LL, RL];
    LMI >= 0;

    rho = param('rho');
    eta_ROA = param('eta_ROA');
    eta_NN = param('eta_NN');
    if param('iter') == 0
        Yk = ones(nu+nphi, nG+nphi);
    else
        Yk = param('Yk');
    end

    % enforce {x: x'Px<=1} \subset {x: |x1| <= x1bound}
     [1,0]*Q1*[1,0]' <= x1bound^2;
    
     % enforce {x: x'Px<=1} \subset {x: |x2| <= x2bound}
     [0,1]*Q1*[0,1]' <= x2bound^2;
    
    % objective function
    obj1 = -eta_ROA*log_det(Q1);
    obj2 = 0.5*rho*pow_pos(norm(fN*Q - L,'fro'), 2);
    obj3 = trace(Yk'*(fN*Q-L));
    obj = obj1 + obj2 + obj3;
    minimize(obj)
cvx_end

% Check properties of Q2
% T_Test_Q2_Properties

%% save computed decision variables (which are reused by python script)
dec_var.Q1 = Q1;
dec_var.Q2 = full(Q2);
dec_var.L1 = L1;
dec_var.L2 = L2;
dec_var.L3 = L3;
dec_var.L4 = L4;
dec_var.Yk = Yk + rho*(fN*Q - L);
dec_var.obj = [obj1, obj2, obj3];

save([param('path') '/' 'sdpvar.mat'], 'dec_var')

%% save NN weights and other useful parameters for result_analysis
weights_and_params = containers.Map;
weights_and_params('AG') = AG;
weights_and_params('BG') = BG;
weights_and_params('x1bound') = x1bound;
weights_and_params('x2bound') = x2bound;
weights_and_params('x_eq') = x_eq;
weights_and_params('eta_NN') = eta_NN;
weights_and_params('eta_ROA') = eta_ROA;
weights_and_params('rho') = rho;
weights_and_params('n_iter') = param('n_iter');
for j = 1:length(n)
    key = sprintf('W%u',j);
    weights_and_params(key) = W(key);
end

save([param('path') '/' 'weights_and_params.mat'], 'weights_and_params')

%% store objective function vector after each iteration
obj4 = param('train_mse');
obj5 = param('test_mse');
unscaled_obj = [obj1/eta_ROA (2/rho)*obj2 obj3 obj4 obj5]; % unscaled objectives where obj2 = ||.||_F^2
scaled_obj = [obj1 (rho/2)*sqrt((2/rho)*obj2) obj3 eta_NN*obj4 obj5] % where obj2 = (rho/2)*||.||_F
filename = [param('logdir') '/' 'obj_data.xlsx'];
if param('iter') == 0
    writematrix(unscaled_obj,filename);
else
    writematrix(unscaled_obj, filename, 'WriteMode', 'append');
end

disp("PARAMETERS, WEIGHTS, AND DATA SAVED")
end