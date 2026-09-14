%% Wastewater Treatment Code
clear
clc
close all

runSVM = true;   % toggle SVM

data = readtable('all_data.csv'); % Loads data as an instance

% Inputs and output
X = [data.Q_inf data.Q_air_1 data.Q_air_2 data.Q_air_3 ...
     data.Q_air_4 data.Q_air_5 data.Temp]; % Input variables that affect the one output we're training
y = data.DO_1;

% Train/test split (80/20)
n = length(y); %n is the vertical length of the data (how many rows)
idx = randperm(n); %makes a random shuffled array of numbers 1 thru n. 
% each of these numbers corresponds to a row in the data
nTrain = round(0.8*n); %takes 80% of this (for training)

trainIdx = idx(1:nTrain); %random rows from data that we're training
testIdx = idx(nTrain+1:end); %random rows from data that we're having it predict

Xtrain = X(trainIdx,:); % input data we're training
yTrain = y(trainIdx); %output data we're training

Xtest = X(testIdx,:); % input data we're testing
yTest = y(testIdx); % output data we're testing

%% Linear Regression
model1 = fitlm(Xtrain,yTrain); %trains linear reg model
yPred1 = predict(model1,Xtest); % model tries to predict data  

rmse1 = sqrt(mean((yPred1 - yTest).^2)); %calculates rmse
R21 = 1 - sum((yTest-yPred1).^2)/sum((yTest-mean(yTest)).^2); % calculates r^2
res1 = yTest - yPred1; % calculates residuals

%% Decision Tree
model2 = fitrtree(Xtrain,yTrain); % trains decision tree model
yPred2 = predict(model2,Xtest); % model tries to predict test data

rmse2 = sqrt(mean((yPred2 - yTest).^2)); % calculates rmse
R22 = 1 - sum((yTest-yPred2).^2)/sum((yTest-mean(yTest)).^2); % calculates r^2
res2 = yTest - yPred2; % calculates residuals

%% SVM (optional)
if runSVM
    model3 = fitrsvm(Xtrain,yTrain); % trains SVM model
    yPred3 = predict(model3,Xtest); % model tries to predict data

    rmse3 = sqrt(mean((yPred3 - yTest).^2)); % calculates rmse
    R23 = 1 - sum((yTest-yPred3).^2)/sum((yTest - mean(yTest)).^2); % calculates r^2
    size(yTest)
    size(yPred3)
    mean(yPred3)
    min(yPred3)
    max(yPred3)
    any(isnan(yPred3))
    res3 = yTest - yPred3; %calculates residuals
end

%% RMSE comparison
if runSVM
    rmseValues = [rmse1 rmse2 rmse3]; % all rmse values including svm
    modelNames = {'Linear','Tree','SVM'}; % model names here
else
    rmseValues = [rmse1 rmse2]; % just decision tree and linear reg rmse
    modelNames = {'Linear','Tree'}; % decision tree and linear names
end

figure % plots rmse values for everything
bar(rmseValues)
set(gca,'XTickLabel',modelNames)
ylabel('RMSE')
title('Model Comparison')

%% Actual vs Predicted plots
figure %plots linear reg 
scatter(yTest,yPred1,'filled')
hold on
plot([min(yTest) max(yTest)],[min(yTest) max(yTest)],'r')
title('Linear Regression')

figure % plots decision tree
scatter(yTest,yPred2,'filled')
hold on
plot([min(yTest) max(yTest)],[min(yTest) max(yTest)],'r')
title('Decision Tree')

if runSVM
    figure % plots svm
    scatter(yTest,yPred3,'filled')
    hold on
    plot([min(yTest) max(yTest)],[min(yTest) max(yTest)],'r')
    title('SVM')
end

%% Residual plots
figure % plots linear residuals
scatter(yPred1,res1,'filled')
yline(0,'r')
title('Linear Residuals')

figure % plots decision tree residuals
scatter(yPred2,res2,'filled')
yline(0,'r')
title('Tree Residuals')

if runSVM
    figure % plots svm residuals
    scatter(yPred3,res3,'filled')
    yline(0,'r')
    title('SVM Residuals')
end

% Display R^2 for each of our models
disp("Linear Regression R^2: "+ R21)
disp("Decision Tree R^2: "+ R22)
if runSVM
disp("SVM R^2: "+ R23);
end
