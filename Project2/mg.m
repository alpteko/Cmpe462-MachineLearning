function model = mg(data,k)
    data_size = length(data);
    r = ones(data_size, k)/2;
    u = cell(k,1);
    cover = cell(k,1);
    prior = cell(k,1);
    %% Initial Variable
    [Ind, means]= kmeans(data,k);
    for i=1:k
        u{i} = means(i,:);
        cover{i} = cov(data(find(Ind == i),:),1);
        prior{i} =  length(find(Ind == i))/data_size;
    end
    while(1)
        %% E- Step
        for i=1:data_size
            for j=1:k
                r(i,j) = prior{j} * mvnpdf(data(i,:),u{j},cover{j});
            end
            r(i,:) = r(i,:)/sum(r(i,:));
        end
        %% M-Step
        old_u = u;
        for i=1:k
            [u{i}, cover{i}, prior{i}] = maximization(data,r(:,i));            
        end
        eps = 0.001;
        err = 0;
        for i=1:k
            err = err + norm(u{i}-old_u{i},1);
        end
        if err < eps
            break;
        end
    end
    model = {u ,cover,prior};
end