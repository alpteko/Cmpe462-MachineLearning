function w = fit(x,y,degree)
observation_len = length(x);
D = zeros(observation_len,degree+1);
for i = 1:observation_len
    for j = 1:degree+1
        D(i,j) = x(i)^(j-1);
    end
end
    w = pinv(D'*D)*D'*y';
end

