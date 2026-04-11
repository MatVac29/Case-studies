function [A,b] = get_covariance_matrix(N,tau,noise)

% This Matlab function helps you obtain a symmetric positive definite
% covariance matrix which can be used with your CG Algorithm.
% It will also produce a rhs vector b 

% Inputs 
% N = Size of matrix A you wish to obtain
% tau = Matern kernel hyper-parameter
% noise = additional added noise

x  = linspace(0,N,N);
[X,Y] = meshgrid(x);
A = (1+sqrt(3)*abs(X-Y)/tau).*exp(-sqrt(3)*abs(X-Y)/tau);
A = (A>10*eps).*A;
A = A + noise*eye(N);
N=size(A,1);
b = randn(N,1);

end