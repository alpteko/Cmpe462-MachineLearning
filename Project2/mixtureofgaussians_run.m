clear all;
load('points2d.dat')
[ ~ , I] = sort(points2d(:,3));
points = points2d(I,:);
p_1 = randperm(2000);
p_2 = randperm(2000) + 2000;
p_3 = randperm(2000) + 4000;
class1 = {points(p_1(1:1000),:),points(p_1(1001:1500),:),points(p_1(1501:2000),:)};
class2 = {points(p_2(1:1000),:),points(p_2(1001:1500),:),points(p_2(1501:2000),:)};
class3 = {points(p_3(1:1000),:),points(p_3(1001:1500),:),points(p_3(1501:2000),:)};
class1_parameters =  cell(3,1);
class2_parameters = cell(3,1);
class3_paramerers = cell(3,1);
valid_set  = [class1{2};class2{2};class3{2}];
test_set  = [class1{3};class2{3};class3{3}];
%% Learn All combinations
for i = 1:3
    class1_parameters{i}= mg(class1{1}(:,1:2),i);
    class2_parameters{i}= mg(class2{1}(:,1:2),i);
    class3_parameters{i} = mg(class3{1}(:,1:2),i);
    parameters{i}={class1_parameters{i} ,class2_parameters{i}, class3_parameters{i}};
end
%% Validate the best combination
maximum = 0;
for i=1:3
    for j=1:3
        for k = 1:3
            class1_model = parameters{i}{1};
            class2_model = parameters{j}{2};
            class3_model = parameters{k}{3};
            p = {class1_model, class2_model, class3_model};
            [acc estimated] = test_mg(valid_set,p,i,j,k);
            if acc > maximum
                maximum = acc;
                valid_clusters = [ i j k];
                valid_p= p;
            end
                
        end
    end
end
%% Test Results
[test_acc estimation ] = test_mg(test_set,valid_p,valid_clusters(1),valid_clusters(2),valid_clusters(3));
valid_clusters
error_rate = 1 - test_acc/1500
confusionmat(test_set(:,3),estimation)