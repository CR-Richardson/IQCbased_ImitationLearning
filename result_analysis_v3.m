% Plots training curves as mean and standard deviation of N repetitions of two experiments.
clear

%% define admin variables and load parameters

% variables to specify - data directory, experiments & repetitions
data_dir = 'PATH TO.../QC-based-IL-main/pendulum_explicit_MPC/data/';
experiment1 = 'size=10_nlayers=2_rho=1_QC=new/size=10_nlayers=2_rho=1_QC=new_';
experiment2 = 'size=10_nlayers=2_rho=1_QC=old/size=10_nlayers=2_rho=1_QC=old_';
exp1_legend = 'Repeated activation';
exp2_legend = 'Combined sector bound';
N = 6; % NUMBER OF REPITITIONS

exp1_color = 'red'; 
exp2_color = 'blue';

% storing experiment directories
exp1_dirs = [];
exp2_dirs = [];
for j=1:N
    new_dir1 = [experiment1 num2str(j) '/'];
    new_dir2 = [experiment2 num2str(j) '/'];
    exp1_dirs = [exp1_dirs; new_dir1];
    exp2_dirs = [exp2_dirs; new_dir2];
end

% load eta_NN, eta_ROA, rho, n_iter
param_dir1 = [exp1_dirs(1,:) num2str(0) '/weights_and_params.mat'];
load(param_dir1, 'weights_and_params');
eta_NN1 = weights_and_params('eta_NN');
eta_ROA1 = weights_and_params('eta_ROA');
rho1 = weights_and_params('rho');

param_dir2 = [exp2_dirs(1,:) num2str(0) '/weights_and_params.mat'];
load(param_dir2, 'weights_and_params');
eta_NN2 = weights_and_params('eta_NN');
eta_ROA2 = weights_and_params('eta_ROA');
rho2 = weights_and_params('rho');

%% load objective function data from excel

obj_data1 = containers.Map; % each entry is data from one repetition
obj_data2 = containers.Map;
NET.addAssembly('microsoft.office.interop.excel');
app = Microsoft.Office.Interop.Excel.ApplicationClass;
for j=1:N
    key = sprintf("%u",j);
    obj_dir = [data_dir exp1_dirs(j,:) 'obj_data.xlsx'];
    book =  app.Workbooks.Open(obj_dir);
    sheet = Microsoft.Office.Interop.Excel.Worksheet(book.Worksheets.Item(1));
    range = sheet.UsedRange;
    arr = range.Value;
    arr = cell(arr,'ConvertTypes',{'all'});
    obj_data1(key) = cell2mat(arr);
    Close(book);
    
    obj_dir = [data_dir exp2_dirs(j,:) 'obj_data.xlsx'];
    book =  app.Workbooks.Open(obj_dir);
    sheet = Microsoft.Office.Interop.Excel.Worksheet(book.Worksheets.Item(1));
    range = sheet.UsedRange;
    arr = range.Value;
    arr = cell(arr,'ConvertTypes',{'all'});
    obj_data2(key) = cell2mat(arr);
    Close(book);
end
Quit(app);

%% Number of objective functions and iterations
no_objs = size(obj_data1('1'),2);
n_iters = size(obj_data1('1'),1);

%% Compute means and standard deviations for each plot

means1 = zeros(n_iters,no_objs);
std_dev1 = zeros(n_iters,no_objs);
means2 = zeros(n_iters,no_objs);
std_dev2 = zeros(n_iters,no_objs);

for i=1:no_objs
    obj1_i = zeros(n_iters,N);
    obj2_i = zeros(n_iters,N);
    for j=1:N
        key = sprintf("%u", j);
        temp1 = obj_data1(key);
        temp2 = obj_data2(key);
        if i==2
            obj1_i(:,j) = sqrt(temp1(:,i));
            obj2_i(:,j) = sqrt(temp2(:,i));
        else    
            obj1_i(:,j) = temp1(:,i);
            obj2_i(:,j) = temp2(:,i);
        end
    end
    means1(:,i) = mean(obj1_i,2);
    std_dev1(:,i) = std(obj1_i,0,2);
    means2(:,i) = mean(obj2_i,2);
    std_dev2(:,i) = std(obj2_i,0,2);
end

%% Compute mean and standard deviation of augmented loss

aug_losses1 = zeros(n_iters,N);
aug_losses2 = zeros(n_iters,N);
for j=1:N
    key = sprintf("%u", j);
    temp1 = obj_data1(key);
    loss = eta_NN1*temp1(:,4) + eta_ROA1*temp1(:,1) + ...
           (rho1/2)*temp1(:,2) + temp1(:,3);
    aug_losses1(:,j) = loss;

    temp1 = obj_data2(key);
    loss = eta_NN2*temp1(:,4) + eta_ROA2*temp1(:,1) + ...
           (rho2/2)*temp1(:,2) + temp1(:,3);
    aug_losses2(:,j) = loss;
end

aug_losses_mean1 = mean(aug_losses1,2);
aug_losses_std1 = std(aug_losses1,0,2);
aug_losses_mean2 = mean(aug_losses2,2);
aug_losses_std2 = std(aug_losses2,0,2);

%% Figure 1 - obj(1) - ROA

x = 1:n_iters;

y1 = means1(x,1)';
z1 = std_dev1(x,1)';
upper_std1 = y1 + z1;
lower_std1 = y1 - z1;

y2 = means2(x,1)';
z2 = std_dev2(x,1)';
upper_std2 = y2 + z2;
lower_std2 = y2 - z2;

if true
    figure(1)
    hold on;
    p1 = plot(x, y1, [exp1_color, '-*']);
    p2 = plot(x, y2, [exp2_color '-*']);
    d1_up = plot(x, upper_std1, [exp1_color ':']);
    d1_low = plot(x, lower_std1, [exp1_color ':']);
    d2_up = plot(x, upper_std2, [exp2_color ':']);
    d2_low = plot(x, lower_std2, [exp2_color ':']);        
    hold off;
    grid minor;
    xlim([1, 12]);
    xlabel('Iteration','interpreter','latex');
    ylabel('$-\log(det(Q1))$','interpreter','latex');
    legend(exp1_legend, exp2_legend, '', '', '', '', 'Location', 'northwest');
    title('ROA', 'interpreter','latex');
end

%% Figure 2 - obj(2) - constraint norm

x = 1:n_iters;

y1 = means1(x,2)';
z1 = std_dev1(x,2)';
upper_std1 = y1 + z1;
lower_std1 = y1 - z1;

y2 = means2(x,2)';
z2 = std_dev2(x,2)';
upper_std2 = y2 + z2;
lower_std2 = y2 - z2;

if true
    figure(2)
    hold on;
    p1 = plot(x, y1, [exp1_color '-*']);
    p2 = plot(x, y2, [exp2_color '-*']);
    d1_up = plot(x, upper_std1, [exp1_color ':']);
    d1_low = plot(x, lower_std1, [exp1_color ':']);
    d2_up = plot(x, upper_std2, [exp2_color ':']);
    d2_low = plot(x, lower_std2, [exp2_color ':']);    
    hold off;
    grid minor;
    xlim([1, 12]);
    ylim([0., 2.5]);
    xlabel('Iteration','interpreter','latex');
    ylabel('$||f(N)Q-L||_{F}$','interpreter','latex');
    legend(exp1_legend, exp2_legend, '', '');
    title('Constraint Deviation', 'interpreter','latex');
end
 
%% Figure 3 - obj(3) - trace

x = 1:n_iters;

y1 = means1(x,3)';
z1 = std_dev1(x,3)';
upper_std1 = y1 + z1;
lower_std1 = y1 - z1;

y2 = means2(x,3)';
z2 = std_dev2(x,3)';
upper_std2 = y2 + z2;
lower_std2 = y2 - z2;

if true
    figure(3)
    hold on;
    p1 = plot(x, y1, [exp1_color '-*']);
    p2 = plot(x, y2, [exp2_color '-*']);
    d1_up = plot(x, upper_std1, [exp1_color ':']);
    d1_low = plot(x, lower_std1, [exp1_color ':']);
    d2_up = plot(x, upper_std2, [exp2_color ':']);
    d2_low = plot(x, lower_std2, [exp2_color ':']);  
    hold off;
    grid minor;
    xlim([1, 12]);
    % ylim([-40., 40.]);
    xlabel('Iteration','interpreter','latex')
    ylabel('$tr(Y^T(f(N)Q-L))$','interpreter','latex')
    legend(exp1_legend, exp2_legend, '', '', 'Location', 'northeast', 'interpreter','latex')
    title('Constraint Deviation Scaled by the Accumulated Constraint Deviation', 'interpreter','latex')
end

%% Figure 4 - obj(4) and obj(5) - MSE 

if true
    figure(4);
    x = 1:n_iters;

    subplot(1,2,1);
    y1 = means1(x,4)';
    z1 = std_dev1(x,4)';
    upper_std1 = y1 + z1;
    lower_std1 = y1 - z1;

    y2 = means2(x,4)';
    z2 = std_dev2(x,4)';
    upper_std2 = y2 + z2;
    lower_std2 = y2 - z2;

    hold on;
    p1 = plot(x, y1, [exp1_color '-*']);
    p2 = plot(x, y2, [exp2_color '-*']);
    d1_up = plot(x, upper_std1, [exp1_color ':']);
    d1_low = plot(x, lower_std1, [exp1_color ':']);
    d2_up = plot(x, upper_std2, [exp2_color ':']);
    d2_low = plot(x, lower_std2, [exp2_color ':']);  
    hold off;
    grid minor;
    xlim([1, 12]);
    xlabel('Iteration','interpreter','latex')
    ylabel('MSE','interpreter','latex')
    legend(exp1_legend, exp2_legend,'', '', 'interpreter','latex')
    title('Training data', 'interpreter','latex')

    subplot(1,2,2)
    y1 = means1(x,5)';
    z1 = std_dev1(x,5)';
    upper_std1 = y1 + z1;
    lower_std1 = y1 - z1;

    y2 = means2(x,5)';
    z2 = std_dev2(x,5)';
    upper_std2 = y2 + z2;
    lower_std2 = y2 - z2;

    hold on;
    p1 = plot(x, y1, [exp1_color '-*']);
    p2 = plot(x, y2, [exp2_color '-*']);
    d1_up = plot(x, upper_std1, [exp1_color ':']);
    d1_low = plot(x, lower_std1, [exp1_color ':']);
    d2_up = plot(x, upper_std2, [exp2_color ':']);
    d2_low = plot(x, lower_std2, [exp2_color ':']);     
    hold off;
    grid minor;
    xlim([1, 12]);
    xlabel('Iteration','interpreter','latex')
    ylabel('MSE','interpreter','latex')
    legend(exp1_legend, exp2_legend,'', '', 'interpreter','latex')
    title('Test data', 'interpreter','latex')

    sgtitle('MSE between NN policy and expert MPC', 'interpreter','latex') % at the end of each iteration
end

%% Figure 5 - Total training obj

x = 1:n_iters;

y1 = aug_losses_mean1';
z1 = aug_losses_std1';
upper_std1 = y1 + z1;
lower_std1 = y1 - z1;

y2 = aug_losses_mean2';
z2 = aug_losses_std2';
upper_std2 = y2 + z2;
lower_std2 = y2 - z2;

if true
    figure(5)
    hold on;
    p1 = plot(x, y1, [exp1_color '-*']);
    p2 = plot(x, y2, [exp2_color '-*']);
    d1_up = plot(x, upper_std1, [exp1_color ':']);
    d1_low = plot(x, lower_std1, [exp1_color ':']);
    d2_up = plot(x, upper_std2, [exp2_color ':']);
    d2_low = plot(x, lower_std2, [exp2_color ':']);  
    hold off;
    grid minor;
    xlim([1, 12]);
    ylim([-40., 0.]);
    xlabel('Iteration','interpreter','latex')
    ylabel('Augmented Loss','interpreter','latex')
    legend(exp1_legend, exp2_legend,'', '', 'Location', 'southeast', 'interpreter','latex')
    title('Augmented Loss', 'interpreter','latex')
end