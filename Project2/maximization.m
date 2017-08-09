function [ u, C, prior ] = maximization(data,r)
    m = sum(r);
    size = length(r);
    prior = m / size;
    u1 = sum(r.*data(:,1))/m;
    u2 = sum(r.*data(:,2))/m;
    u = [u1 u2];
    E = squeeze(data);
    C = zeros(2,2);
    for i=1:size
       C = C + r(i)*(E(i,:)-u)'* (E(i,:)-u);
    end 
    C = C/m;
end

