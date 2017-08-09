clear all;
load('points2d.dat')
[ ~ , I] = sort(points2d(:,3));
points = points2d(I,:);
p_1 = randperm(2000);
p_2 = randperm(2000) + 2000;
p_3 = randperm(2000) + 4000;
labeled_set = [ points(p_1(1:1000),:) ; points(p_2(1:1000),:); points(p_3(1:1000),:)];
valid_set  = [ points(p_1(1001:1500),:) ; points(p_2(1001:1500),:); points(p_3(1001:1500),:)];
test_set  = [ points(p_1(1501:2000),:) ; points(p_2(1501:2000),:); points(p_3(1501:2000),:)];
%% Validation for k
tp_test = zeros(3,1);
k_i = 1;
for k=[1,10,40]
    for i=1:1500
        p = valid_set(i,:);
        class_estimate = knn_estimate(p(1:2),labeled_set,k);
        if p(3) == class_estimate
            tp_test(k_i) = tp_test(k_i) + 1; 
        end
    end
    k_i = k_i +1;
end
%% Test
tp = zeros(3,1);
k_i = 1;
class_estimates = zeros(3,1500);
for k=[1,10,40]
    for i=1:1500
        p = test_set(i,:);
        class_estimates(k_i,i) = knn_estimate(p(1:2),labeled_set,k);
        if p(3) == class_estimates(k_i,i)
            tp(k_i) = tp(k_i) + 1; 
        end
    end
    k_i = k_i +1;
end

%% k=1
disp('k=1')
1- tp(1)/1500
confusionmat(test_set(:,3),class_estimates(1,:))
%% k=10
disp('k=10')
1- tp(2)/1500
confusionmat(test_set(:,3),class_estimates(2,:))
%% k=40
disp('k=40')
1- tp(3)/1500
confusionmat(test_set(:,3),class_estimates(3,:))