function [acc,estimated_class] = test_mg( data, models,cn1,cn2,cn3)
data_size = length(data);
acc = 0;
estimated_class = zeros(data_size,1);
for i=1:data_size
    p1 = 0;
    p2 = 0;
    p3 = 0;
    point = data(i,1:2);
    real_class = data(i,3);
    model1_parameters = models{1};
    for j=1:cn1
         u = model1_parameters{1}{j};
         cover = model1_parameters{2}{j};
         prior = model1_parameters{3}{j};
         p1 = p1+ prior * mvnpdf(point,u,cover);
    end
    model2_parameters = models{2};
    for j=1:cn2
         u = model2_parameters{1}{j};
         cover = model2_parameters{2}{j};
         prior = model2_parameters{3}{j};
         p2 = p2+ prior * mvnpdf(point,u,cover);
    end
    model3_parameters = models{3};
    for j=1:cn3
         u = model3_parameters{1}{j};
         cover = model3_parameters{2}{j};
         prior = model3_parameters{3}{j};
         p3 = p3+ prior * mvnpdf(point,u,cover);
    end
    [~, ind] = max([p1 p2 p3]);
    estimated_class(i) = ind-1;
    if real_class == estimated_class(i)
        acc = acc+1;
    end

end


end

