% Kernel ridge regression with a fixed polynomial kernel of (1 + <x1,x2>)^d.
% User supplies y, x, grid of lambda (penalty weights).
% RMSE of one-step-ahead out-of-sample forecast is used to select the optimal lambda. 


function [lambda_star, alpha_star, rmse_vec, K] = kernel_ridge_reg_fixed_poly(y, x, oos_percent, d, lambda_vec)
% Inputs:
%   y: a n-by-1 vector of target (don't need to demean);
%   x: a n-by-m matrix of features;
%   oos_percent: a scalar of the percent of out-of-sample validation data (e.g. 0.1 -> 10%);
%   d: a scalar of the order of polynomial (e.g. d = 3 for cubic polynomial);
%   lambda_vec: a k-by-1 vector of the lambda grid;
% Outputs:
%   lambda_star: a scalar of the optimal lambda;
%   alpha_star: a n-by-1 vector of the optimal kernel weights;
%   rmse_vec: a k-by-1 vector of the RMSE in out-of-sample forecasts for each element in lambda grid;
%   K: a n-by-n matrix of the in-sample kernel matrix.

y_demeaned = y - mean(y);
nobs = length(y_demeaned);
nobs_pred = round(nobs * oos_percent);
nobs_est = nobs - nobs_pred;

%% Over a grid of lambda (penalty weight), compute the RMSE of one-step-ahead out-of-sample forecasts
nof_lambda = length(lambda_vec);
rmse_vec = zeros(nof_lambda,1);
for li = 1:nof_lambda
    lambda = lambda_vec(li);
    
    % Compute RMSE of one-step-ahead out-of-sample forecast
    se_vec = zeros(nobs_pred,1);
    for t = 1:nobs_pred
        % Select the right estimation sample
        nobs_est_t = nobs_est+t-1; 
        y_iter_est = y_demeaned(1:nobs_est_t);
        x_iter_est = x(1:nobs_est_t,:);

        % Compute in-sample kernel matrix
        K = (1 + x_iter_est * x_iter_est') .^ d;

        % Given a value of lambda, compute the optimal kernel weights
        tmp_mat = K + lambda * eye(nobs_est_t);
        tmp_mat_inv = tmp_mat\eye(nobs_est_t);
        alpha = tmp_mat_inv * y_iter_est;

        % Compute SE on out-of-sample data
        y_t = y_demeaned(nobs_est_t+1);
        x_t = x(nobs_est_t+1,:)';
        k_t = (1 + x_iter_est * x_t) .^ d;
        y_t_model = alpha' * k_t;
        se_vec(t) = (y_t - y_t_model)^2;
    end
    
    rmse_vec(li) = sqrt(mean(se_vec));
end


%% Find the optimal lambda
[~, idx] = min(rmse_vec);
lambda_star = lambda_vec(idx);


%% Re-estimate the optimal kernel weight based on full data
K = (1 + x * x') .^ d;
tmp_mat = K + lambda_star * eye(nobs);
tmp_mat_inv = tmp_mat\eye(nobs);
alpha_star = tmp_mat_inv * y_demeaned;
    
    




        






