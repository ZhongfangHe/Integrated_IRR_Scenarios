% Compute PC of interest rates;
% Kernel ridge regression of each macro variable on the first 3 PC and the lags of these 3 PC;
%
% Use a fixed polynomial kernel (1+<x,y>)^d

clear;

%% Inputs
function_path = 'C:\Users\Documents\Research\Machine Learning\Toolboxes\Self_Written_Functions\kernel_ridge_regression'; 

read_path = 'C:\Users\Documents\Projects\Behavior Models\Integrated Risk Scenario\Macro Variables\';
read_file = [read_path, 'CAD_Macroeconomic.xlsx'];
read_sheet_rates = 'InterestRates';
read_sheet_macro = 'MacroVariables'; %both sheets have the same start/end dates

read_row_start = 111; %row in spread sheet to start reading data (2000 Jan)
read_row_end = 339; %row in spread sheet to stop reading data (most recent month)

span_inflation = 12; %span of inflation rate in month
span_stock = 12; %avg log stock return over "span" months
span_housing = 12; %avg log housing return over "span" months

oos_percent = 0.3; % percent of out-of-sample data to select optimal penalty weight 
d = 3; %order of polynomial kernel
lambda_vec{1} = 14e6:1e5:16e6; %grid of penalty weights for GDP
lambda_vec{2} = [10   100  1000   1500:100:2500  3000   4000   5000]; %grid of penalty weights for unemployment
lambda_vec{3} = [10   100  1000  2000  3000:100:5000  6000   7000]; %grid of penalty weights for inflation
lambda_vec{4} = [10  100  500  900:100:1500  2000  3000]; %grid of penalty weights for stock
lambda_vec{5} = [100  200  300  400  500:10:700  800  900 1000]; %grid of penalty weights for housing

pc_lag = 3; %max number of PC lags in regression (pc_lag = 3 --> pc(-1), pc(-2), pc(-3))



%% Read interest rate data
read_cell1 = ['B',num2str(read_row_start),':O', num2str(read_row_end)];
read_cell2 = ['Q',num2str(read_row_start),':T', num2str(read_row_end)];
rates1 = xlsread(read_file, read_sheet_rates, read_cell1);
rates2 = xlsread(read_file, read_sheet_rates, read_cell2);
rates = [rates1  rates2];
clear rates1  rates2;


%% Read macroeconomic data, compute factors
read_cell = ['B',num2str(read_row_start - span_inflation), ':F', num2str(read_row_end)];
raw = xlsread(read_file, read_sheet_macro, read_cell);
nobs_raw = size(raw,1);
gdp =  raw((1+span_inflation):nobs_raw,1);
unemployment =  raw((1+span_inflation):nobs_raw,2);
inflation = 100 * (raw((1+span_inflation):nobs_raw,3) ./ raw(1:(nobs_raw-span_inflation),3) - 1);
stock = 100 * log(raw((1+span_inflation):nobs_raw,4) ./ raw((1+span_inflation-span_stock):(nobs_raw-span_stock),4)) /span_stock;
housing = 100 * log(raw((1+span_inflation):nobs_raw,5) ./ raw((1+span_inflation-span_housing):(nobs_raw-span_housing),5)) /span_housing;
factors = [gdp  unemployment  inflation  stock  housing];
factors_name = {'GDP','UnemploymentRate','InflationRate',['StockReturn(',num2str(span_stock),'MAvg)'],['HousingReturn(',num2str(span_housing),'MAvg)']};
[nobs,nof_factors] = size(factors);

write_file = read_file;
xlswrite(write_file, factors, 'Factors', 'B2');  


%% Compute PC of interest rates
rates_mean = mean(rates);
[coeff, score, latent] = princomp(rates); % (data - mean(data)) * coeff = score
var_explained = latent./sum(latent);

nof_pc_selected = 3; %select only first 3 components
pc_selected = score(:,1:nof_pc_selected);
pc_selected_name = {'PC1','PC2','PC3'};
const = -1 * rates_mean * coeff(:,1:nof_pc_selected);
write_sheet_pc = 'PrinComp';
xlswrite(write_file, pc_selected, write_sheet_pc, 'B2');
xlswrite(write_file, [coeff(:,1:nof_pc_selected); const], write_sheet_pc, 'G2');
xlswrite(write_file, var_explained(1:nof_pc_selected), write_sheet_pc, 'L2');
xlswrite(write_file, rates * coeff(:,1:nof_pc_selected) + kron(const, ones(nobs,1)), write_sheet_pc, 'N2'); %Reconstruct PC to validate


%% Regression of macro variables on PC
addpath(genpath(function_path)); %add the path of the functions

nof_coef = nof_pc_selected * (1 + pc_lag);
reg_coef = zeros(nobs-pc_lag+3, nof_factors); %alpha, mean(y), R-square, stdev of residual

fitted_macro = zeros(nobs-pc_lag,nof_factors);

x = zeros(nobs-pc_lag, nof_coef);
count = 1;
for i = 1:(1+pc_lag)
    x(:,(i*nof_pc_selected-2):(i*nof_pc_selected)) = pc_selected((pc_lag-i+2):(nobs-i+1),:);
end
lambda_star_vec = zeros(nof_factors,1);
rmse_cell = cell(nof_factors,1);
tic;
for fi = 1:nof_factors
    y = factors((pc_lag+1):nobs,fi);
    
    [lambda_star_vec(fi), alpha_star, rmse_cell{fi}, K] = kernel_ridge_reg_fixed_poly(y, x, oos_percent, d, lambda_vec{fi});
    
    fitted_macro(:,fi) = mean(y) + K * alpha_star;
    resid = y - fitted_macro(:,fi);
    stdev_resid = std(resid);
    R_square = 1 - var(resid)/var(y);
    reg_coef(:,fi) = [alpha_star; mean(y); R_square; stdev_resid];
    
    disp(factors_name{fi});
    toc;
end
write_sheet_reg = 'Kernel_Lag';
xlswrite(write_file, reg_coef, write_sheet_reg, 'B2');

    
xlswrite(write_file, fitted_macro, 'Factor_Analysis_Kernel', ['H',num2str(2+pc_lag)]);

%% Check the optimal lambdas
for fi = 1:nof_factors
    subplot(3,2,fi);
    plot(lambda_vec{fi}, rmse_cell{fi}, 'o-');
    title(factors_name{fi});
end
subplot(3,2,6);
plot(lambda_star_vec, 'o-');
title('Optimal Lambda');
saveas(gcf,'Optimal_Lambda.fig');
save('kernel_rmse.mat', 'rmse_cell', 'lambda_vec', 'lambda_star_vec','factors_name');



