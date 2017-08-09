function class_estimate = knn_estimate(p,data,k)
    data_size = length(data);
    distances = zeros(data_size,1);
    for i=1:data_size
        distances(i) = norm(data(i,1:2)-p);
    end
    temp = max(distances);
    min_list = zeros(k,1);
    %% For efficiency not call sort
    if k<11
        for i =1:k
            [~, index] = min(distances);
            distances(index) = temp;
            min_list(i) = data(index,3); 
        end
    else
        [ ~, sorted] = sort(distances);
        index_list = sorted(1:k);
         min_list = data(index_list,3);     
    end
    class_estimate = mode(min_list);

end

