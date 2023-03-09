% Plots mean and standard deviation across training for varying NN depth
clear


%% define admin variables and load parameters

% variables to specify - data directory, repetitions, & experiments
data_dir = 'PATH TO.../QC-based-IL-main/pendulum_explicit_MPC/data/';
N = 6; % NUMBER OF REPETITIONS

% new QC
new_legend = 'Repeated activation';
new_color = 'red'; 
experiment1 = 'size=10_nlayers=1_rho=10_QC=new/size=10_nlayers=1_rho=10_QC=new_';
experiment2 = 'size=10_nlayers=2_rho=10_QC=new/size=10_nlayers=2_rho=10_QC=new_';
experiment3 = 'size=10_nlayers=3_rho=10_QC=new/size=10_nlayers=3_rho=10_QC=new_';
experiment4 = 'size=10_nlayers=4_rho=10_QC=new/size=10_nlayers=4_rho=10_QC=new_';
experiment5 = 'size=10_nlayers=5_rho=10_QC=new/size=10_nlayers=5_rho=10_QC=new_';

% old QC
old_legend = 'Combined sector bound';
old_color = 'blue';
experiment6 = 'size=10_nlayers=1_rho=10_QC=old/size=10_nlayers=1_rho=10_QC=old_';
experiment7 = 'size=10_nlayers=2_rho=10_QC=old/size=10_nlayers=2_rho=10_QC=old_';
experiment8 = 'size=10_nlayers=3_rho=10_QC=old/size=10_nlayers=3_rho=10_QC=old_';
experiment9 = 'size=10_nlayers=4_rho=10_QC=old/size=10_nlayers=4_rho=10_QC=old_';
experiment10 = 'size=10_nlayers=5_rho=10_QC=old/size=10_nlayers=5_rho=10_QC=old_';

% storing experiment directories
exp1_dirs = []; exp2_dirs = []; exp3_dirs = []; exp4_dirs = []; exp5_dirs = [];
exp6_dirs = []; exp7_dirs = []; exp8_dirs = []; exp9_dirs = []; exp10_dirs = [];
for j=1:N
    new_dir1 = [experiment1 num2str(j) '/'];
    new_dir2 = [experiment2 num2str(j) '/'];
    new_dir3 = [experiment3 num2str(j) '/'];
    new_dir4 = [experiment4 num2str(j) '/'];
    new_dir5 = [experiment5 num2str(j) '/'];
    new_dir6 = [experiment6 num2str(j) '/'];
    new_dir7 = [experiment7 num2str(j) '/'];
    new_dir8 = [experiment8 num2str(j) '/'];
    new_dir9 = [experiment9 num2str(j) '/'];
    new_dir10 = [experiment10 num2str(j) '/'];
    
    exp1_dirs = [exp1_dirs; new_dir1];
    exp2_dirs = [exp2_dirs; new_dir2];
    exp3_dirs = [exp3_dirs; new_dir3];
    exp4_dirs = [exp4_dirs; new_dir4];
    exp5_dirs = [exp5_dirs; new_dir5];
    exp6_dirs = [exp6_dirs; new_dir6];
    exp7_dirs = [exp7_dirs; new_dir7];
    exp8_dirs = [exp8_dirs; new_dir8];
    exp9_dirs = [exp9_dirs; new_dir9];
    exp10_dirs = [exp10_dirs; new_dir10];
end

% load eta_NN, eta_ROA, rho - same for all experiments
param_dir = [exp1_dirs(1,:) num2str(0) '/weights_and_params.mat'];
load(param_dir, 'weights_and_params');
eta_NN = weights_and_params('eta_NN');
eta_ROA = weights_and_params('eta_ROA');
rho = weights_and_params('rho');

%% load objective function data from excel

obj_data1 = containers.Map; % each entry is data from one repetition
obj_data2 = containers.Map;
obj_data3 = containers.Map;
obj_data4 = containers.Map;
obj_data5 = containers.Map;
obj_data6 = containers.Map;
obj_data7 = containers.Map;
obj_data8 = containers.Map;
obj_data9 = containers.Map;
obj_data10 = containers.Map;
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

    obj_dir = [data_dir exp3_dirs(j,:) 'obj_data.xlsx'];
    book =  app.Workbooks.Open(obj_dir);
    sheet = Microsoft.Office.Interop.Excel.Worksheet(book.Worksheets.Item(1));
    range = sheet.UsedRange;
    arr = range.Value;
    arr = cell(arr,'ConvertTypes',{'all'});
    obj_data3(key) = cell2mat(arr);
    Close(book);

    obj_dir = [data_dir exp4_dirs(j,:) 'obj_data.xlsx'];
    book =  app.Workbooks.Open(obj_dir);
    sheet = Microsoft.Office.Interop.Excel.Worksheet(book.Worksheets.Item(1));
    range = sheet.UsedRange;
    arr = range.Value;
    arr = cell(arr,'ConvertTypes',{'all'});
    obj_data4(key) = cell2mat(arr);
    Close(book);

    obj_dir = [data_dir exp5_dirs(j,:) 'obj_data.xlsx'];
    book =  app.Workbooks.Open(obj_dir);
    sheet = Microsoft.Office.Interop.Excel.Worksheet(book.Worksheets.Item(1));
    range = sheet.UsedRange;
    arr = range.Value;
    arr = cell(arr,'ConvertTypes',{'all'});
    obj_data5(key) = cell2mat(arr);
    Close(book);

    obj_dir = [data_dir exp6_dirs(j,:) 'obj_data.xlsx'];
    book =  app.Workbooks.Open(obj_dir);
    sheet = Microsoft.Office.Interop.Excel.Worksheet(book.Worksheets.Item(1));
    range = sheet.UsedRange;
    arr = range.Value;
    arr = cell(arr,'ConvertTypes',{'all'});
    obj_data6(key) = cell2mat(arr);
    Close(book);

    obj_dir = [data_dir exp7_dirs(j,:) 'obj_data.xlsx'];
    book =  app.Workbooks.Open(obj_dir);
    sheet = Microsoft.Office.Interop.Excel.Worksheet(book.Worksheets.Item(1));
    range = sheet.UsedRange;
    arr = range.Value;
    arr = cell(arr,'ConvertTypes',{'all'});
    obj_data7(key) = cell2mat(arr);
    Close(book);

    obj_dir = [data_dir exp8_dirs(j,:) 'obj_data.xlsx'];
    book =  app.Workbooks.Open(obj_dir);
    sheet = Microsoft.Office.Interop.Excel.Worksheet(book.Worksheets.Item(1));
    range = sheet.UsedRange;
    arr = range.Value;
    arr = cell(arr,'ConvertTypes',{'all'});
    obj_data8(key) = cell2mat(arr);
    Close(book);

    obj_dir = [data_dir exp9_dirs(j,:) 'obj_data.xlsx'];
    book =  app.Workbooks.Open(obj_dir);
    sheet = Microsoft.Office.Interop.Excel.Worksheet(book.Worksheets.Item(1));
    range = sheet.UsedRange;
    arr = range.Value;
    arr = cell(arr,'ConvertTypes',{'all'});
    obj_data9(key) = cell2mat(arr);
    Close(book);

    obj_dir = [data_dir exp10_dirs(j,:) 'obj_data.xlsx'];
    book =  app.Workbooks.Open(obj_dir);
    sheet = Microsoft.Office.Interop.Excel.Worksheet(book.Worksheets.Item(1));
    range = sheet.UsedRange;
    arr = range.Value;
    arr = cell(arr,'ConvertTypes',{'all'});
    obj_data10(key) = cell2mat(arr);
    Close(book);
end
Quit(app);

%% Number of objective functions and iterations
no_objs = size(obj_data1('1'),2);
n_iters = size(obj_data1('1'),1);

%% Compute means and standard deviations for each plot

means1 = zeros(n_iters,no_objs);
means2 = zeros(n_iters,no_objs);
means3 = zeros(n_iters,no_objs);
means4 = zeros(n_iters,no_objs);
means5 = zeros(n_iters,no_objs);
means6 = zeros(n_iters,no_objs);
means7 = zeros(n_iters,no_objs);
means8 = zeros(n_iters,no_objs);
means9 = zeros(n_iters,no_objs);
means10 = zeros(n_iters,no_objs);
std_dev1 = zeros(n_iters,no_objs);
std_dev2 = zeros(n_iters,no_objs);
std_dev3 = zeros(n_iters,no_objs);
std_dev4 = zeros(n_iters,no_objs);
std_dev5 = zeros(n_iters,no_objs);
std_dev6 = zeros(n_iters,no_objs);
std_dev7 = zeros(n_iters,no_objs);
std_dev8 = zeros(n_iters,no_objs);
std_dev9 = zeros(n_iters,no_objs);
std_dev10 = zeros(n_iters,no_objs);

for i=1:no_objs
    obj1_i = zeros(n_iters,N);
    obj2_i = zeros(n_iters,N);
    obj3_i = zeros(n_iters,N);
    obj4_i = zeros(n_iters,N);
    obj5_i = zeros(n_iters,N);
    obj6_i = zeros(n_iters,N);
    obj7_i = zeros(n_iters,N);
    obj8_i = zeros(n_iters,N);
    obj9_i = zeros(n_iters,N);
    obj10_i = zeros(n_iters,N);
    for j=1:N
        key = sprintf("%u", j);
        temp1 = obj_data1(key);
        temp2 = obj_data2(key);
        temp3 = obj_data3(key);
        temp4 = obj_data4(key);
        temp5 = obj_data5(key);
        temp6 = obj_data6(key);
        temp7 = obj_data7(key);
        temp8 = obj_data8(key);
        temp9 = obj_data9(key);
        temp10 = obj_data10(key);
        if i==2
            obj1_i(:,j) = sqrt(temp1(:,i));
            obj2_i(:,j) = sqrt(temp2(:,i));
            obj3_i(:,j) = sqrt(temp3(:,i));
            obj4_i(:,j) = sqrt(temp4(:,i));
            obj5_i(:,j) = sqrt(temp5(:,i));
            obj6_i(:,j) = sqrt(temp6(:,i));
            obj7_i(:,j) = sqrt(temp7(:,i));
            obj8_i(:,j) = sqrt(temp8(:,i));
            obj9_i(:,j) = sqrt(temp9(:,i));
            obj10_i(:,j) = sqrt(temp10(:,i));
        else    
            obj1_i(:,j) = temp1(:,i);
            obj2_i(:,j) = temp2(:,i);
            obj3_i(:,j) = temp3(:,i);
            obj4_i(:,j) = temp4(:,i);
            obj5_i(:,j) = temp5(:,i);
            obj6_i(:,j) = temp6(:,i);
            obj7_i(:,j) = temp7(:,i);
            obj8_i(:,j) = temp8(:,i);
            obj9_i(:,j) = temp9(:,i);
            obj10_i(:,j) = temp10(:,i);
        end
    end

    % mean of each iteration across N experiments
    means1(:,i) = mean(obj1_i,2);
    means2(:,i) = mean(obj2_i,2);
    means3(:,i) = mean(obj3_i,2);
    means4(:,i) = mean(obj4_i,2);
    means5(:,i) = mean(obj5_i,2);
    means6(:,i) = mean(obj6_i,2);
    means7(:,i) = mean(obj7_i,2);
    means8(:,i) = mean(obj8_i,2);
    means9(:,i) = mean(obj9_i,2);
    means10(:,i) = mean(obj10_i,2);
    
    % std dev of each iteration across N experiments
    std_dev1(:,i) = std(obj1_i,0,2);
    std_dev2(:,i) = std(obj2_i,0,2);
    std_dev3(:,i) = std(obj3_i,0,2);
    std_dev4(:,i) = std(obj4_i,0,2);
    std_dev5(:,i) = std(obj5_i,0,2);
    std_dev6(:,i) = std(obj6_i,0,2);
    std_dev7(:,i) = std(obj7_i,0,2);
    std_dev8(:,i) = std(obj8_i,0,2);
    std_dev9(:,i) = std(obj9_i,0,2);
    std_dev10(:,i) = std(obj10_i,0,2);
end

% mean of the iteration (2-12) means 
means_5n = mean(means1(2:12,:));
means_10n = mean(means2(2:12,:));
means_15n = mean(means3(2:12,:));
means_20n = mean(means4(2:12,:));
means_25n = mean(means5(2:12,:));
means_5o = mean(means6(2:12,:));
means_10o = mean(means7(2:12,:));
means_15o = mean(means8(2:12,:));
means_20o = mean(means9(2:12,:));
means_25o = mean(means10(2:12,:));

% std dev of the iteration (2-12) means
std_dev_5n = std(means1(2:12,:));
std_dev_10n = std(means2(2:12,:));
std_dev_15n = std(means3(2:12,:));
std_dev_20n = std(means4(2:12,:));
std_dev_25n = std(means5(2:12,:));
std_dev_5o = std(means6(2:12,:));
std_dev_10o = std(means7(2:12,:));
std_dev_15o = std(means8(2:12,:));
std_dev_20o = std(means9(2:12,:));
std_dev_25o = std(means10(2:12,:));

%% Figure 1 - obj(1) - ROA

x = [1, 2, 3, 4, 5]; % depth of NN

y_new = [means_5n(1), means_10n(1), means_15n(1), means_20n(1), means_25n(1)];
z_new = [std_dev_5n(1), std_dev_10n(1), std_dev_15n(1), std_dev_20n(1), std_dev_25n(1)];
upper_std_new = y_new + z_new;
lower_std_new = y_new - z_new;

y_old = [means_5o(1), means_10o(1), means_15o(1), means_20o(1), means_25o(1)];
z_old = [std_dev_5o(1), std_dev_10o(1), std_dev_15o(1), std_dev_20o(1), std_dev_25o(1)];
upper_std_old = y_old + z_old;
lower_std_old = y_old - z_old;

if true
    figure(1)
    hold on;
    p_new = plot(x, y_new, [new_color '-*']); 
    p_old = plot(x, y_old, [old_color '-*']);
    d_new_up = plot(x, upper_std_new, [new_color ':']);
    d_new_low = plot(x, lower_std_new, [new_color ':']);
    d_old_up = plot(x, upper_std_old, [old_color ':']);
    d_old_low = plot(x, lower_std_old, [old_color ':']);
    hold off;
    grid minor;
    xlim([1, 5]);
    xlabel('Number of hidden layers','interpreter','latex');
    ylabel('$-\log(det(Q1))$','interpreter','latex');
    legend(new_legend, old_legend,'', '', 'Location', 'southeast');
    title('ROA', 'interpreter','latex');
end

%% Figure 2 - obj(2) - constraint norm

x = [1, 2, 3, 4, 5]; % depth of NN

y_new = [means_5n(2), means_10n(2), means_15n(2), means_20n(2), means_25n(2)];
z_new = [std_dev_5n(2), std_dev_10n(2), std_dev_15n(2), std_dev_20n(2), std_dev_25n(2)];
upper_std_new = y_new + z_new;
lower_std_new = y_new - z_new;

y_old = [means_5o(2), means_10o(2), means_15o(2), means_20o(2), means_25o(2)];
z_old = [std_dev_5o(2), std_dev_10o(2), std_dev_15o(2), std_dev_20o(2), std_dev_25o(2)];
upper_std_old = y_old + z_old;
lower_std_old = y_old - z_old;

if true
    figure(2)
    hold on;
    p_new = plot(x, y_new, [new_color '-*']);
    p_old = plot(x, y_old, [old_color '-*']);
    d_new_up = plot(x, upper_std_new, [new_color ':']);
    d_new_low = plot(x, lower_std_new, [new_color ':']);
    d_old_up = plot(x, upper_std_old, [old_color ':']);
    d_old_low = plot(x, lower_std_old, [old_color ':']);    
    hold off;
    grid minor;
    xlim([1, 5]);
    % ylim([-1., 5.]);
    xlabel('Number of hidden layers','interpreter','latex');
    ylabel('$||f(N)Q-L||_{F}$','interpreter','latex');
    legend(new_legend, old_legend, '', '');
    title('Constraint Deviation', 'interpreter','latex');
end

%% Figure 3 - obj(4) and obj(5) - MSE 

if true
    figure(3);
    x = [1, 2, 3, 4, 5];

    subplot(1,2,1);
    y_new = [means_5n(4), means_10n(4), means_15n(4), means_20n(4), means_25n(4)];
    z_new = [std_dev_5n(4), std_dev_10n(4), std_dev_15n(4), std_dev_20n(4), std_dev_25n(4)];
    upper_std_new = y_new + z_new;
    lower_std_new = y_new - z_new;

    y_old = [means_5o(4), means_10o(4), means_15o(4), means_20o(4), means_25o(4)];
    z_old = [std_dev_5o(4), std_dev_10o(4), std_dev_15o(4), std_dev_20o(4), std_dev_25o(4)];
    upper_std_old = y_old + z_old;
    lower_std_old = y_old - z_old;

    hold on;
    p_new = plot(x, y_new, [new_color '-*']);
    p_old = plot(x, y_old, [old_color '-*']);
    d_new_up = plot(x, upper_std_new, [new_color ':']);
    d_new_low = plot(x, lower_std_new, [new_color ':']);
    d_old_up = plot(x, upper_std_old, [old_color ':']);
    d_old_low = plot(x, lower_std_old, [old_color ':']);
    hold off;
    grid minor;
    xlim([1, 5]);
    xlabel('Number of hidden layers','interpreter','latex')
    ylabel('MSE','interpreter','latex')
    legend(new_legend, old_legend, '', '', 'interpreter','latex')
    title('Training data', 'interpreter','latex')

    subplot(1,2,2)
    y_new = [means_5n(5), means_10n(5), means_15n(5), means_20n(5), means_25n(5)];
    z_new = [std_dev_5n(5), std_dev_10n(5), std_dev_15n(5), std_dev_20n(5), std_dev_25n(5)];
    upper_std_new = y_new + z_new;
    lower_std_new = y_new - z_new;

    y_old = [means_5o(5), means_10o(5), means_15o(5), means_20o(5), means_25o(5)];
    z_old = [std_dev_5o(5), std_dev_10o(5), std_dev_15o(5), std_dev_20o(5), std_dev_25o(5)];
    upper_std_old = y_old + z_old;
    lower_std_old = y_old - z_old;

    hold on;
    p_new = plot(x, y_new, [new_color '-*']);
    p_old = plot(x, y_old, [old_color '-*']);
    d_new_up = plot(x, upper_std_new, [new_color ':']);
    d_new_low = plot(x, lower_std_new, [new_color ':']);
    d_old_up = plot(x, upper_std_old, [old_color ':']);
    d_old_low = plot(x, lower_std_old, [old_color ':']);
    hold off;
    grid minor;
    xlim([1, 5]);
    xlabel('Number of hidden layers','interpreter','latex')
    ylabel('MSE','interpreter','latex')
    legend(new_legend, old_legend,'', '', 'interpreter','latex')
    title('Test data', 'interpreter','latex')

    sgtitle('MSE between NN policy and expert MPC', 'interpreter','latex') % at the end of each iteration
end
