function [u ,q] = estimator(x)
N = length(x);
u = sum(x)/N;
q = sqrt(sum((x - u).^2)/(N));
end