function y  = poly(x,w)
y = zeros(1,length(x));
degree = length(w);
for i = 1:degree
    y = y + x.^(i-1)*w(i);
end
end