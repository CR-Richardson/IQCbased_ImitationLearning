% Plots comparing ROA from different experiments
clear

% Comparison experiments, iteration number & legend captions
data_dir = 'PATH TO.../QC-based-IL-main/pendulum_explicit_MPC/data/';
exp1_no = 3; exp2_no = 3;
iter1 = 9; iter2 = 10;
exp1 = sprintf('size=10_nlayers=2_rho=1_QC=new/size=10_nlayers=2_rho=1_QC=new_%u/',exp1_no);
exp2 = sprintf('size=10_nlayers=2_rho=1_QC=old/size=10_nlayers=2_rho=1_QC=old_%u/',exp2_no);
exp1_legend = 'RA';
exp2_legend = 'CSB';
exp1_color = 'maroon';
exp2_color = 'coolblue';
mpc_color = 'orange';
poly_color = 'darkgray';

%% Controller comparison and ROA

% ****************** Controller comparison ******************
figure(1)
subplot(1,2,1)
% MPC
load('exp_data2.csv')
p1 = plot3(exp_data2(:,2),exp_data2(:,1),exp_data2(:,3),'ro','MarkerSize',2);
p1.Color = mycolor(mpc_color);
set(gca,'YDir','reverse');
hold on

% NN policy #1
folder = exp1;
iter = iter1;
Load_params;

xU = [x1bound; x2bound];
xL = -xU;
num_pt = 50;
x1 = linspace(xL(1), xU(1),num_pt);
x2 = linspace(xL(2),xU(2),num_pt);
ug = zeros(num_pt,num_pt);
for exp1_no=1:num_pt
    for exp2_no = 1:num_pt
        ug(exp2_no,exp1_no) = nn_eval(W,b,[x1(exp1_no); x2(exp2_no)]);
    end
end
h1 = mesh(x2,x1,ug);
h1.FaceAlpha = 0.5;
h1.FaceColor = mycolor(exp1_color);
h1.EdgeColor = mycolor(exp1_color);
h1.EdgeAlpha  = 0.4;

% NN policy #2
folder = exp2;
iter = iter2;
Load_params;
xU = [x1bound; x2bound];
xL = -xU;
num_pt = 50;
x1 = linspace(xL(1), xU(1),num_pt);
x2 = linspace(xL(2),xU(2),num_pt);
ug = zeros(num_pt,num_pt);
for exp1_no=1:num_pt
    for exp2_no = 1:num_pt
        ug(exp2_no,exp1_no) = nn_eval(W,b,[x1(exp1_no); x2(exp2_no)]);
    end
end
h2 = mesh(x2,x1,ug);
h2.FaceAlpha = 0.5;
h2.FaceColor = mycolor(exp2_color);
h2.EdgeColor = mycolor(exp2_color);
h2.EdgeAlpha  = 0.4;

grid on
xlabel('$x_2$','interpreter','latex')
ylabel('$x_1$','interpreter','latex')
zlabel('$u$','interpreter','latex')
legend('MPC', exp1_legend, exp2_legend, 'interpreter','latex')
t1 = title('NN vs. expert MPC');
t1.FontWeight = 'bold';
t1.FontSize = 20;
hold off

% % ************************ ROA ************************
subplot(1,2,2)

% MPC
p2= plot(exp_data2(:,1),exp_data2(:,2),'ro','MarkerSize',2);
p2.Color = mycolor(mpc_color);
hold on

% plot state constraint set 
X = Polyhedron('lb',[-x1bound; -x2bound],'ub',[x1bound; x2bound]);
X.plot('alpha',0.4,'color',mycolor(poly_color),'linewidth',3,'edgecolor',mycolor(poly_color))

% NN policy #1
folder = exp1;
iter = iter1;
Load_params;
ROA_NN(Q1, exp1_color);

% NN policy #2
folder = exp2;
iter = iter2;
Load_params;
ROA_NN(Q1,exp2_color);

grid on;
axis([-x1bound-0.5 x1bound+0.5 -x2bound-1 x2bound+1]);
xlabel('$x_1$','interpreter','latex')
ylabel('$x_2$','interpreter','latex')
legend('MPC', 'X', exp1_legend, exp2_legend,'interpreter','latex')
t2 = title('ROA');
t2.FontWeight = 'bold';
t2.FontSize = 20;
hold off

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

%%
function ROA_NN(Q1,color)
% plot ROA
pvar x1 x2
V = [x1,x2]*inv(Q1)*[x1;x2];
domain1 = [-5, 5, -10, 10];
[C,h] = pcontour(V,1,domain1,'r');
h.LineColor = mycolor(color);
h.LineWidth = 3;
end
