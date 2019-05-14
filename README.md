# Integrated_IRR_Scenarios
Project macroeconomic variables that are consistent with interest rate projections from prescribed interest rate scenarios.

The data of client behavior regression is in the spreadsheet "Example_Data.xlsx", while the data of macroeconomic and interest rate variables are in the spreadsheet "CAD_Macroeconomic.xlsx".

The Matlab file "main_kernel_macro_lags.m" estimates the conditional expectation of each macroeconomic variable on interest rates (first 3 PC and their 3 lags) via KRR (the function "kernel_ridge_reg_fixed_poly.m"). The kernel is a cubic polynomial kernel.
