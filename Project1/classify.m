%%%%%%%%%%
%% Case 3
load('iris.mat');
setosa = iris.features(1:50,:);
versicolour = iris.features(51:100,:);
R = zeros(4,1);
Monte_Carlo = 200;
%% For most efficient feature 
%% Monte Carlo Simulation is applied.
for k = 1:Monte_Carlo
%% Traninng and Test Set is separated.
p = randperm(50);
K = 40;
L = 50 - K;
training = p(1:K);
test = p(K+1:50);
u1 = zeros(1,4);
u2 = zeros(1,4);
q1 = zeros(1,4);
q2 = zeros(1,4);
    %% For each feature parameter estimation is done.
    for i=1:4
        %%
        [u1(i),q1(i)] = estimator(setosa(training,i)); 
        [u2(i),q2(i)] = estimator(versicolour(training,i));
        t_set = [setosa(test,i)',versicolour(test,i)'];
        %% Relative Likelihood is calculated
        %% and compared to classidy.
        p_setosa = normpdf(t_set,u1(i),q1(i));
        p_versicolour = normpdf(t_set,u2(i),q2(i));
        %% Setosa is Labeled as 0.
        %% Versicolour is labeled as 1.
        results = p_setosa < p_versicolour;
        true =[zeros(1,L),ones(1,L)];
        compare = true - results;
        R(i) = R(i) + length(find(compare == 0)) /(2*L);
    end
end
R =  R / Monte_Carlo;
figure;
plot(1,R(1),'O','markersize',10);
hold on;
plot(2,R(2),'O','markersize',10);
hold on;
plot(3,R(3),'O','markersize',10);
hold on;
plot(4,R(4),'O','markersize',10);
hold on;
legend('SepalLength','SepalWidth','PetalLength','PetalWidth');
