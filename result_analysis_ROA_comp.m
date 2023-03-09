% Plots multiple cherry picked ROA from a selection of experiments
clear

%% Comparison experiments, iteration number & legend captions
data_dir = 'PATH TO.../QC-based-IL-main/pendulum_explicit_MPC/data/';
poly_color = 'darkgray';
mpc_color = 'orange';

% Legends
exp1_legend = '5';
exp2_legend = '10';
exp3_legend = '15';
exp4_legend = '20';
exp5_legend = '25';

% Best repeated activation experiments
exp1_no = 5; exp2_no = 6; exp3_no = 5; exp4_no = 4; exp5_no = 5;
iter1 = 7; iter2 = 11; iter3 = 10; iter4 = 5; iter5 = 5;

exp1 = sprintf('size=5_nlayers=2_rho=10_QC=new/size=5_nlayers=2_rho=10_QC=new_%u/',exp1_no);
exp2 = sprintf('size=10_nlayers=2_rho=10_QC=new/size=10_nlayers=2_rho=10_QC=new_%u/',exp2_no);
exp3 = sprintf('size=15_nlayers=2_rho=10_QC=new/size=15_nlayers=2_rho=10_QC=new_%u/',exp3_no);
exp4 = sprintf('size=20_nlayers=2_rho=10_QC=new/size=20_nlayers=2_rho=10_QC=new_%u/',exp4_no);
exp5 = sprintf('size=25_nlayers=2_rho=10_QC=new/size=25_nlayers=2_rho=10_QC=new_%u/',exp5_no);

% Best combined sector bound experiments
exp6_no = 4; exp7_no = 2; exp8_no = 5; exp9_no = 1; exp10_no = 2;
iter6 = 9; iter7 = 11; iter8 = 11; iter9 = 2; iter10 = 9;

exp6 = sprintf('size=5_nlayers=2_rho=10_QC=old/size=5_nlayers=2_rho=10_QC=old_%u/',exp6_no);
exp7 = sprintf('size=10_nlayers=2_rho=10_QC=old/size=10_nlayers=2_rho=10_QC=old_%u/',exp7_no);
exp8 = sprintf('size=15_nlayers=2_rho=10_QC=old/size=15_nlayers=2_rho=10_QC=old_%u/',exp8_no);
exp9 = sprintf('size=20_nlayers=2_rho=10_QC=old/size=20_nlayers=2_rho=10_QC=old_%u/',exp9_no);
exp10 = sprintf('size=25_nlayers=2_rho=10_QC=old/size=25_nlayers=2_rho=10_QC=old_%u/',exp10_no);

%% ************************ ROA Repeated activation ************************
figure(1);
subplot(1,2,1)

% initialising variables
folder = exp1;
iter = iter1;
Load_params;
domain1 = [-5, 5, -10, 10];

% MPC
load('exp_data2.csv')
hold on;
p1= plot(exp_data2(:,1),exp_data2(:,2),'ro','MarkerSize',2);
p1.Color = mycolor(mpc_color);


% plot state constraint set 
X = Polyhedron('lb',[-x1bound; -x2bound],'ub',[x1bound; x2bound]);
X.plot('alpha',0.4,'color',mycolor(poly_color),'linewidth',3,'edgecolor',mycolor(poly_color))

% NN policy #1
folder = exp1;
iter = iter1;
Load_params;
pvar x1 x2
V = [x1,x2]*inv(Q1)*[x1;x2];
[C1,h1] = pcontour(V,1,domain1);
h1.LineWidth = 3;
h1.LineColor = 'blue';

% NN policy #2
folder = exp2;
iter = iter2;
Load_params;
pvar x1 x2
V = [x1,x2]*inv(Q1)*[x1;x2];
[C2,h2] = pcontour(V,1,domain1);
h2.LineWidth = 3;
h2.LineColor = 'red';

% NN policy #3
folder = exp3;
iter = iter3;
Load_params;
pvar x1 x2
V = [x1,x2]*inv(Q1)*[x1;x2];
[C3,h3] = pcontour(V,1,domain1);
h3.LineWidth = 3;
h3.LineColor = 'magenta';

% NN policy #4
folder = exp4;
iter = iter4;
Load_params;
pvar x1 x2
V = [x1,x2]*inv(Q1)*[x1;x2];
[C4,h4] = pcontour(V,1,domain1);
h4.LineWidth = 3;
h4.LineColor = 'cyan';

% NN policy #5
folder = exp5;
iter = iter5;
Load_params;
pvar x1 x2
V = [x1,x2]*inv(Q1)*[x1;x2];
[C5,h5] = pcontour(V,1,domain1);
h5.LineWidth = 3;
h5.LineColor = 'green';

grid on;
axis([-x1bound-0.5 x1bound+0.5 -x2bound-1 x2bound+1]);
xlabel('$x_1$','interpreter','latex')
ylabel('$x_2$','interpreter','latex')
legend('', '', exp1_legend, exp2_legend, exp3_legend, exp4_legend, exp5_legend, 'interpreter','latex')
t1 = title('Repeated Activation', 'interpreter', 'latex');
t1.FontWeight = 'bold';
t1.FontSize = 20;
hold off;

%% ************************ ROA Combined sector bound ************************
subplot(1,2,2)
 
% MPC
load('exp_data2.csv')
hold on;
p2= plot(exp_data2(:,1),exp_data2(:,2),'ro','MarkerSize',2);
p2.Color = mycolor(mpc_color);


% plot state constraint set 
X = Polyhedron('lb',[-x1bound; -x2bound],'ub',[x1bound; x2bound]);
X.plot('alpha',0.4,'color',mycolor(poly_color),'linewidth',3,'edgecolor',mycolor(poly_color))

% NN policy #6
folder = exp6;
iter = iter6;
Load_params;
pvar x1 x2
V = [x1,x2]*inv(Q1)*[x1;x2];
[C6,h6] = pcontour(V,1,domain1);
h6.LineWidth = 3;
h6.LineColor = 'blue';

% NN policy #7
folder = exp7;
iter = iter7;
Load_params;
pvar x1 x2
V = [x1,x2]*inv(Q1)*[x1;x2];
[C7,h7] = pcontour(V,1,domain1);
h7.LineWidth = 3;
h7.LineColor = 'red';

% NN policy #8
folder = exp8;
iter = iter8;
Load_params;
pvar x1 x2
V = [x1,x2]*inv(Q1)*[x1;x2];
[C8,h8] = pcontour(V,1,domain1);
h8.LineWidth = 3;
h8.LineColor = 'magenta';

% NN policy #9
folder = exp9;
iter = iter9;
Load_params;
pvar x1 x2
V = [x1,x2]*inv(Q1)*[x1;x2];
[C9,h9] = pcontour(V,1,domain1);
h9.LineWidth = 3;
h9.LineColor = 'cyan';

% NN policy #10
folder = exp10;
iter = iter10;
Load_params;
pvar x1 x2
V = [x1,x2]*inv(Q1)*[x1;x2];
[C10,h10] = pcontour(V,1,domain1);
h10.LineWidth = 3;
h10.LineColor = 'green';

grid on;
axis([-x1bound-0.5 x1bound+0.5 -x2bound-1 x2bound+1]);
xlabel('$x_1$','interpreter','latex')
ylabel('$x_2$','interpreter','latex')
legend('', '', exp1_legend, exp2_legend, exp3_legend, exp4_legend, exp5_legend, 'interpreter','latex')
t2 = title('Combined Sector Bound', 'interpreter', 'latex');
t2.FontWeight = 'bold';
t2.FontSize = 20;
hold off;

%%
sgt = sgtitle('ROA for NN with Varying Width', 'interpreter', 'latex');
sgt.FontWeight = 'bold';
sgt.FontSize = 26;
%%
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
