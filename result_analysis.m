% Plots training curves, control law and ROA comparison with MPC law. 
% Plots are of a single repetition of a single experiment.

clear

%% load data and model parameters

% variables to specify - folder name and iteration number
data_dir = 'PATH TO.../QC-based-IL-main/pendulum_explicit_MPC/data/';
folder = '/'; % EXPERIMENT
n_iter = ; % ITERATION O COMPARE TO MPC CONTROL LAW AND ROA

fname = [folder num2str(n_iter) '/sdpvar.mat'];
fname2 = [folder num2str(n_iter) '/weights_and_params.mat'];
fname3 = [data_dir folder 'obj_data.xlsx'];

load(fname2,'weights_and_params')
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

x_eq = weights_and_params('x_eq');
x1bound = weights_and_params('x1bound');
x2bound = weights_and_params('x2bound');
AG = weights_and_params('AG');
BG = weights_and_params('BG');
nG = size(AG, 1);
eta_NN = weights_and_params('eta_NN');
eta_ROA = weights_and_params('eta_ROA');
rho = weights_and_params('rho');

load(fname, 'dec_var')
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
N_diff = fN-L*inv(Q);

%% Figure 1 - Final Constraint difference
if true
    figure(1)
    subplot(2,2,1)
    h1 = mesh(fN);
    h1.FaceAlpha = 0.5;
    h1.FaceColor = mycolor('coolblue');
    h1.EdgeColor = mycolor('coolblue');
    h1.EdgeAlpha  = 0.5;
    h1.Marker = '.';
    zlim([-1.6, 0.65])
    view([20, 20, 5])
    title('mesh$(f(N))$','interpreter','latex')
    garyfyFigure
    subplot(2,2,2)
    h2 = mesh(L*inv(Q));
    h2.FaceAlpha = 0.5;
    h2.FaceColor = mycolor('orange');
    h2.EdgeColor = mycolor('orange');
    h2.EdgeAlpha  = 0.5;
    h2.Marker = '.';
    zlim([-1.6, 0.65])
    view([20, 20, 5])
    title('mesh$(LQ^{-1})$','interpreter','latex')
    subplot(2,2,3)
    h3 = mesh(fN*Q);
    h3.FaceAlpha = 0.5;
    h3.FaceColor = mycolor('coolblue');
    h3.EdgeColor = mycolor('coolblue');
    h3.EdgeAlpha  = 0.5;
    h3.Marker = '.';
    zlim([-1.6, 0.65])
    view([20, 20, 5])
    title('mesh$(f(N)*Q)$','interpreter','latex')
    garyfyFigure
    subplot(2,2,4)
    h4 = mesh(L);
    h4.FaceAlpha = 0.5;
    h4.FaceColor = mycolor('orange');
    h4.EdgeColor = mycolor('orange');
    h4.EdgeAlpha  = 0.5;
    h4.Marker = '.';
    zlim([-1.6, 0.65])
    view([20, 20, 5])
    title('mesh$(L)$','interpreter','latex')
    garyfyFigure
end

%% Figure 2 - NN policy, ROA
xU = [x1bound; x2bound];
xL = -xU;
num_pt = 50;
x1 = linspace(xL(1), xU(1),num_pt);
x2 = linspace(xL(2),xU(2),num_pt);
ug = zeros(num_pt,num_pt);
for i=1:num_pt
    for j = 1:num_pt
        ug(j,i) = nn_eval(W,b,[x1(i); x2(j)]);
    end
end
figure(2)
subplot(1,2,1)
h = mesh(x2,x1,ug);
h.FaceAlpha = 0.5;
h.FaceColor = mycolor('coolblue');
h.EdgeColor = mycolor('coolblue');
h.EdgeAlpha  = 0.4;
hold on
load('exp_data2.csv')
p1 = plot3(exp_data2(:,2),exp_data2(:,1),exp_data2(:,3),'ro','MarkerSize',2);
p1.Color = mycolor('orange');
set(gca,'YDir','reverse');
grid on
xlabel('$x_2$','interpreter','latex')
ylabel('$x_1$','interpreter','latex')
zlabel('$u$','interpreter','latex')
legend('NN policy', 'MPC', 'interpreter','latex')
title('NN policy vs. Expert MPC')
garyfyFigure

% ROA
subplot(1,2,2)
p2= plot(exp_data2(:,1),exp_data2(:,2),'ro','MarkerSize',2);
p2.Color = mycolor('orange'); % removed for NN policy # 2
hold on
ROA_NN(fNvx, fNvw, fNux, fNuw, nG, nphi,AG,BG,x1bound,x2bound,Q1)
legend('MPC','$X$', 'NN policy' ,'interpreter','latex')
title('ROA')
garyfyFigure

%% load in data from excel
NET.addAssembly('microsoft.office.interop.excel');
app = Microsoft.Office.Interop.Excel.ApplicationClass;
book =  app.Workbooks.Open(fname3);
sheet = Microsoft.Office.Interop.Excel.Worksheet(book.Worksheets.Item(1)); 
range = sheet.UsedRange;
arr = range.Value;
obj_data = cell(arr,'ConvertTypes',{'all'});
obj_data = cell2mat(obj_data);
Close(book);
Quit(app);

%% Figure 3 - obj(1) - ROA

if true
    figure(3)
    plot(0:n_iter,obj_data(1:n_iter+1,1),'-*')
    grid minor
    xlabel('Iteration','interpreter','latex')
    ylabel('$-\log(det(Q1))$','interpreter','latex')
    legend('NN policy', 'interpreter','latex')
    title('ROA', 'interpreter','latex')
end

%% Figure 4 - obj(2) - constraint norm

if true
    figure(4)
    semilogy(0:n_iter,sqrt(obj_data(1:n_iter+1,2)),'-*')
    hold on
    semilogy(0:n_iter, 0.5*ones(length(0:n_iter)),'r')
    hold off
    grid minor
    xlabel('Iteration','interpreter','latex')
    ylabel('$||f(N)Q-L||_{F}$','interpreter','latex')
    legend('NN policy', '$0.5$', 'interpreter','latex')
    title('Constraint Deviation', 'interpreter','latex')
end

%% Figure 5 - obj(3) - trace

if true
    figure(5)
    plot(0:n_iter,obj_data(1:n_iter+1,3),'-*')
    grid minor
    xlabel('Iteration','interpreter','latex')
    ylabel('$tr(Y^T(f(N)Q-L))$','interpreter','latex')
    legend('NN policy', 'interpreter','latex')
    title('Constraint Deviation Scaled by the Accumulated Constraint Deviation', 'interpreter','latex')
end

%% Figure 6 - obj(4) and obj(5) - MSE 

if true
    figure(6)
    subplot(1,2,1)
    semilogy(0:n_iter,obj_data(1:n_iter+1,4),'-*')
    grid minor
    xlabel('Iteration','interpreter','latex')
    ylabel('MSE','interpreter','latex')
    legend('NN policy', 'interpreter','latex')
    title('Training data', 'interpreter','latex')

    subplot(1,2,2)
    semilogy(0:n_iter,obj_data(1:n_iter+1,5),'-*')
    grid minor
    xlabel('Iteration','interpreter','latex')
    ylabel('MSE','interpreter','latex')
    legend('NN policy', 'interpreter','latex')
    title('Test data', 'interpreter','latex')

    sgtitle('MSE between NN policy and expert MPC', 'interpreter','latex') % at the end of each iteration
end

%% Figure 7 - Total training obj


aug_loss = eta_NN*obj_data(:,4) + eta_ROA*obj_data(:,1) + ...
(2/rho)*obj_data(:,2) + obj_data(:,3);

if true
    figure(7)
    plot(0:n_iter,aug_loss(1:n_iter+1),'-*')
    grid minor
    xlabel('Iteration','interpreter','latex')
    ylabel('Augmented Loss','interpreter','latex')
    legend('NN policy', 'interpreter','latex')
    title('Augmented Loss', 'interpreter','latex')
end

%% Useful functions

function u = nn_eval(W,b,x)
    z = x;
    for i = 1:length(W)-1
        key_W = sprintf('W%u',i);
        key_b = sprintf('b%u',i);
        z = W(key_W)*z + b(key_b);
        z = tanh(z);
    end
    key_W = sprintf('W%u',i+1);
    key_b = sprintf('b%u',i+1);
    u = W(key_W)*z + b(key_b);
end

function ROA_NN(fNvx, fNvw, fNux, fNuw,nG,nphi,AG,BG,x1bound,x2bound,Q1)
% plot state constraint set 
X = Polyhedron('lb',[-x1bound; -x2bound],'ub',[x1bound; x2bound]);
X.plot('alpha',0.4,'color',mycolor('lightgray'),'linewidth',3,'edgecolor',mycolor('darkgray'))
hold on

% plot ROA
pvar x1 x2
V = [x1,x2]*inv(Q1)*[x1;x2];
domain1 = [-5, 5, -10, 10];
[C,h] = pcontour(V,1,domain1,'r');
h.LineColor = mycolor('coolblue');
h.LineWidth = 3;

grid on;
axis([-x1bound-0.5 x1bound+0.5 -x2bound-1 x2bound+1]);
xlabel('$x_1$','interpreter','latex')
ylabel('$x_2$','interpreter','latex')
end