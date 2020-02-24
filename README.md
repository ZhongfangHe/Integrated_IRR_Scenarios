# Integrated_IRR_Scenarios
Integrate macroeconomic variables into behavior models for interest rate risk (IRR) measurement of the banking book such that projections of macroeconomic variables are consistent with prescribed IRR interest rate scenarios.

The files can be used to replicate the example in my paper "Integrating macroeconomic variables with interest rate scenarios for interest rate risk measurement in the banking book" (May 2019).

The data of client behavior regression is in the spreadsheet "Example_Data.xlsx", while the data of macroeconomic and interest rate variables are in the spreadsheet "CAD_Macroeconomic.xlsx".

The Matlab file "main_kernel_macro_lags.m" estimates the conditional expectation of each macroeconomic variable on interest rates (first 3 PC and their 3 lags) via kernel ridge regression (the function "kernel_ridge_reg_fixed_poly.m"). The kernel is a simplified cubic polynomial kernel.
