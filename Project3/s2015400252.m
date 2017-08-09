%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Cmpe 462 HW 3            %%
%% Author : Alptekin Orbay  %%
%% Student ID : 2015400252  %%
%%                          %%
%%                          %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Closing All Plots
%% Clearing All variables
%% Loading Data
close all;
clear;
load('points2d.dat');
%% Initial weights and biases
first_weights = rand(5,2)-0.5;
first_bias = rand(5,1)-0.5;
second_weights = rand(3,5)-0.5;
second_bias = rand(3,1)-0.5;

%% Train and Test Set is arranged.
len = length(points2d);
points2d = points2d(randperm(len),:);
test_set = points2d(1:50,:);
points2d = points2d(51:400,:);
%gscatter(points2d(:,1),points2d(:,2),points2d(:,3))
%% Variable Declarations 
len = length(points2d);
first_layer = zeros(len,5);
second_layer = zeros(len,3);
d1_w = zeros(5,2);
d2_w = zeros(3,5);
d1_b = zeros(5,1);
d2_b = zeros(3,1);
alpha = 0.15;
y = zeros(len,3);
count = 0;
train_error = zeros(500,1);
test_error = zeros(500,1);


%% Training Part
for ep=1:500
    %% For every Epoch Online Weight Update
    % Random instance picking
    points2d = points2d(randperm(len),:);
    %% For each instance calculate error and update weights
    for i=1:len
        %% Calculating Output
        for j=1:5
            first_layer(i,j) = perceptron(points2d(i,1:2),first_weights(j,:),first_bias(j));
        end
        for j=1:3
            second_layer(i,j) = softmax(first_layer(i,:),second_weights(j,:),second_bias(j));
        end
        for j=1:3
           y(i,j) = exp(second_layer(i,j));
        end
        
        %% Calculating Error
        y(i,:) = y(i,:)/ sum(y(i,:));
        r = zeros(3,1);
        r(points2d(i,3)+1) = 1;
        for j=1:3
            train_error(ep) = train_error(ep) - (r(j) *log(y(i,j)));
        end
        %% Hidden Layer Output weight gradient 
        for j=1:3
            r = 0 * ones(3,1);
            r(points2d(i,3)+1) = 1;
            %% Weight Gradient
            d2_w(j,:) = alpha * (r(j)-y(i,j)) * first_layer(i,:);
            %% Bias Gradient
            d2_b(j) = alpha * (r(j)-y(i,j));
        end
         
        %% ?nput Hidden Layer weight gradient
        for j=1:5
            r = 0 * ones(1,3);
            r(points2d(i,3)+1) = 1;
            e = 0;
            for k=1:3
                e = e + (r(k) - y(i,k))* second_weights(k,j);
            end
            %% Weight Gradient
            d1_w(j,:) = alpha * e *(1-first_layer(i,j)^2) * points2d(i,1:2);
            %% Bias Gradient
            d1_b(j) = alpha * e*(1-first_layer(i,j)^2);
        end
        
        %% Updating The Weights and Biases
         first_weights = first_weights + d1_w;
         second_weights = second_weights + d2_w; 
         first_bias = first_bias + d1_b;
         second_bias = second_bias + d2_b;
       
    end
    %% Every Epoch Learning Rate will be reduced.
    alpha = alpha * 0.99;
    
    %% In every epoch test error calculated.
    for i=1:length(test_set)
        %% Classify Test Set
        for j=1:5
            first_layer(i,j) = perceptron(test_set(i,1:2),first_weights(j,:),first_bias(j));
        end
        for j=1:3
            second_layer(i,j) = softmax(first_layer(i,:),second_weights(j,:),second_bias(j));
        end
        for j=1:3
           y(i,j) = exp(second_layer(i,j));
        end
        %% Error Calculate
        y(i,:) = y(i,:)/ sum(y(i,:));
        r = 0 * ones(3,1);
        r(test_set(i,3)+1) = 1;
        for j=1:3
            test_error(ep) = test_error(ep) + -1  * r(j) *log(y(i,j));
        end    
    end
   
end

%% Plotting Train and Test Errors in traning 
figure;
plot(train_error/350)
hold on;
plot(test_error/50)
legend('TrainError','TestError')

%% appling trained MLP onto Test Set
for i=1:length(test_set)
        for j=1:5
            first_layer(i,j) = perceptron(test_set(i,1:2),first_weights(j,:),first_bias(j));
        end
        for j=1:3
            second_layer(i,j) = softmax(first_layer(i,:),second_weights(j,:),second_bias(j));
        end
        for j=1:3
           y(i,j) = exp(second_layer(i,j));
        end
        y(i,:) = y(i,:)/ sum(y(i,:));
end
test_results = zeros(length(test_set),1);

%% Evaluating MLP
for i=1:length(test_set)
    [~,ind] = max(y(i,:));
    if ind-1 == test_set(i,3)
        test_results(i) = ind-1;
    else
        test_results(i) = ind-1+3;
    end
end
figure;
gscatter(test_set(:,1),test_set(:,2),test_results)
hold on;
gscatter(test_set(i:1),test_set(:,2),test_set(:,3))

%% Calculating Confusion Matrix
for i=1:length(test_set)
    [~,ind] = max(y(i,:));
    test_results(i) = ind-1;
end
confusion_matrix = confusionmat(test_set(:,3),test_results)

%% true positive negative , false positive negative
for c=0:2
    tp = zeros(length(test_set),1);
    fp = zeros(length(test_set),1);
    tn = zeros(length(test_set),1);
    fn = zeros(length(test_set),1);
    for i=1:length(test_set)
        if test_set(i,3) == test_results(i)
            if test_results(i) ==  c
                tp(i) = 1;
            else
                tn(i) = 1;
            end
        else
            if test_set(i,3) == c
                fn(i) = 1;
            elseif test_results(i) == c
                fp(i) = 1;
            else
                tn(i) = 1;
            end
        end
    end
    figure;
    title(strcat('Results for',' ',num2str(c)));
        hold on;
    plot(test_set(tp==1,1),test_set(tp==1,2),'*')
        hold on;
    plot(test_set(fp==1,1),test_set(fp==1,2),'*')
        hold on;
    plot(test_set(tn==1,1),test_set(tn==1,2),'*')
        hold on;
    plot(test_set(fn==1,1),test_set(fn==1,2),'*')
        hold on;
    legend('TruePositive','FalsePositive','TrueNegative','FalseNegative');
end

%% Plotting Boundaries
[X1, Y1 ] = meshgrid(-12:0.1:8,-12:0.1:8);
color(:,1) = X1(:);
color(:,2) = Y1(:); 
for i=1:length(color)
        for j=1:5
            first_layer(i,j) = perceptron(color(i,1:2),first_weights(j,:),first_bias(j));
        end
        for j=1:3
            second_layer(i,j) = softmax(first_layer(i,:),second_weights(j,:),second_bias(j));
        end
        for j=1:3
           y(i,j) = exp(second_layer(i,j));
        end
        y(i,:) = y(i,:)/ sum(y(i,:));
end
results = zeros(length(color),1);
for i=1:length(color)
    [~,ind] = max(y(i,:));
    results(i) = ind-1;
end
figure;
gscatter(color(:,1),color(:,2),results)

%% Helper Functions
function p = perceptron(x,w,b)
    X = w * x';
    p = tanh(X+b);
end

function X = softmax(x,w,b)
        X = w * x' +b;
end
