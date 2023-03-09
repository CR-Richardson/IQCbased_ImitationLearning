% Plots training curves as mean and standard deviation of N repetitions of a single experiment.
clear

%% define admin variables and load parameters

% variables to specify - data directory, experiment & repetitions
data_dir = 'PATH TO.../QC-based-IL-main/pendulum_explicit_MPC/data/';
experiment = 'size=10_nlayers=2_rho=1_QC=old_';
N = 3; % NUMBER OF REPETITIONS
mean_color = 'red'; 
std_color = '#FF7276';

% storing experiment directories
exp_dirs = [];
for j=1:N
    new_dir = [experiment num2str(j) '/'];
    exp_dirs = [exp_dirs; new_dir];
end

% load eta_NN, eta_ROA, rho, n_iter
param_dir = [exp_dirs(1,:) num2str(0) '/weights_and_params.mat'];
load(param_dir, 'weights_and_params');
eta_NN = weights_and_params('eta_NN');
eta_ROA = weights_and_params('eta_ROA');
rho = weights_and_params('rho');

%% load objective function data from excel
obj_data = containers.Map;
NET.addAssembly('microsoft.office.interop.excel');
app = Microsoft.Office.Interop.Excel.ApplicationClass;
for j=1:N
    obj_dir = [data_dir exp_dirs(j,:) 'obj_data.xlsx'];
    key = sprintf("%u",j);
    book =  app.Workbooks.Open(obj_dir);
    sheet = Microsoft.Office.Interop.Excel.Worksheet(book.Worksheets.Item(1)); 
    range = sheet.UsedRange;
    arr = range.Value;
    arr = cell(arr,'ConvertTypes',{'all'});
    obj_data(key) = cell2mat(arr);
    Close(book);
end
Quit(app);

%% Number of objective functions and iterations
no_objs = size(obj_data('1'),2);
n_iters = size(obj_data('1'),1);

%% Compute means and standard deviations for each plot

means = zeros(n_iters,no_objs);
std_dev = zeros(n_iters,no_objs);
for i=1:no_objs
    obj_i = zeros(n_iters,N);
    for j=1:N
        key = sprintf("%u", j);
        temp = obj_data(key);
        if i==2 % stores ||.||_F rather than ||.||_F^2
            obj_i(:,j) = sqrt(temp(:,i));
        else
            obj_i(:,j) = temp(:,i);
        end
    end
    means(:,i) = mean(obj_i,2);
    std_dev(:,i) = std(obj_i,0,2);
end

%% Compute mean and standard deviation of augmented loss

aug_losses = zeros(n_iters,N);
for j=1:N
    key = sprintf("%u", j);
    temp = obj_data(key);
    loss = eta_NN*temp(:,4) + eta_ROA*temp(:,1) + ...
           (rho/2)*temp(:,2) + temp(:,3);
    aug_losses(:,j) = loss;
end

aug_losses_mean = mean(aug_losses,2);
aug_losses_std = std(aug_losses,0,2);

%% Figure 1 - obj(1) - ROA

x = 1:n_iters;
y = means(x,1)';
z = std_dev(x,1)';
upper_std = y + z;
lower_std = y - z;
x2 = [x, fliplr(x)];
inBetween = [upper_std, fliplr(lower_std)];

if true
    figure(1)
    f = fill(x2, inBetween, mean_color);
    f.FaceColor = std_color;
    hold on;
    plot(x, y, [mean_color '-*']);
    hold off;
    grid minor;
    xlim([1, 12]);
    xlabel('Iteration','interpreter','latex');
    ylabel('$-\log(det(Q1))$','interpreter','latex');
    legend('', 'Sample Mean', 'Location', 'northwest');
    title('ROA', 'interpreter','latex');
end

%% Figure 2 - obj(2) - constraint norm

x = 1:n_iters;
y = means(x,2)';
z = std_dev(x,2)';
upper_std = y + z;
lower_std = y - z;
x2 = [x, fliplr(x)];
inBetween = [upper_std, fliplr(lower_std)];

if true
    figure(2)
    f = fill(x2, inBetween, mean_color);
    f.FaceColor = std_color;
    hold on;
    plot(x, y, [mean_color '-*']);
    hold off;
    grid minor;
    xlim([1, 12]);
    ylim([-3., 10.]);
    xlabel('Iteration','interpreter','latex');
    ylabel('$||f(N)Q-L||_{F}^2$','interpreter','latex');
    legend('', 'Sample Mean', 'interpreter','latex');
    title('Constraint Deviation', 'interpreter','latex');
end

%% Figure 3 - obj(3) - trace

x = 1:n_iters;
y = means(x,3)';
z = std_dev(x,3)';
upper_std = y + z;
lower_std = y - z;
x2 = [x, fliplr(x)];
inBetween = [upper_std, fliplr(lower_std)];

if true
    figure(3)
    f = fill(x2, inBetween, mean_color);
    f.FaceColor = std_color;
    hold on;
    plot(x, y, [mean_color '-*']);
    hold off;
    grid minor;
    xlim([1, 12]);
    ylim([-10., 10.]);
    xlabel('Iteration','interpreter','latex')
    ylabel('$tr(Y^T(f(N)Q-L))$','interpreter','latex')
    legend('', 'Sample Mean', 'Location', 'northeast', 'interpreter','latex')
    title('Constraint Deviation Scaled by the Accumulated Constraint Deviation', 'interpreter','latex')
end

%% Figure 4 - obj(4) and obj(5) - MSE 

if true
    figure(4);
    subplot(1,2,1);

    x = 1:n_iters;
    y = means(x,4)';
    z = std_dev(x,4)';
    upper_std = y + z;
    lower_std = y - z;
    x2 = [x, fliplr(x)];
    inBetween = [upper_std, fliplr(lower_std)];

    f = fill(x2, inBetween, mean_color);
    f.FaceColor = std_color;
    hold on;
    plot(x, y, [mean_color '-*']);
    hold off;
    grid minor;
    xlim([1, 12]);
    xlabel('Iteration','interpreter','latex')
    ylabel('MSE','interpreter','latex')
    legend('', 'Sample Mean', 'interpreter','latex')
    title('Training data', 'interpreter','latex')

    subplot(1,2,2)
    
    x = 1:n_iters;
    y = means(x,5)';
    z = std_dev(x,5)';
    upper_std = y + z;
    lower_std = y - z;
    x2 = [x, fliplr(x)];
    inBetween = [upper_std, fliplr(lower_std)];

    f = fill(x2, inBetween, mean_color);
    f.FaceColor = std_color;
    hold on;
    plot(x, y, [mean_color '-*']);
    hold off;
    grid minor;
    xlim([1, 12]);
    xlabel('Iteration','interpreter','latex')
    ylabel('MSE','interpreter','latex')
    legend('', 'Sample Mean', 'interpreter','latex')
    title('Test data', 'interpreter','latex')

    sgtitle('MSE between NN policy and expert MPC', 'interpreter','latex') % at the end of each iteration
end

%% Figure 5 - Total training obj

x = 1:n_iters;
y = aug_losses_mean';
z = aug_losses_std';
upper_std = y + z;
lower_std = y - z;
x2 = [x, fliplr(x)];
inBetween = [upper_std, fliplr(lower_std)];

if true
    figure(5)
    f = fill(x2, inBetween, mean_color);
    f.FaceColor = std_color;
    hold on;
    plot(x, y, [mean_color '-*']);
    hold off;
    grid minor;
    xlim([1, 12]);
    ylim([-50., -5.]);
    xlabel('Iteration','interpreter','latex')
    ylabel('Augmented Loss','interpreter','latex')
    legend('', 'Sample Mean', 'Location', 'southeast', 'interpreter','latex')
    title('Augmented Loss', 'interpreter','latex')
end