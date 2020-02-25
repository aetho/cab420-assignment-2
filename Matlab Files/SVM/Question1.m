%% CAB420 - Machine Learning 
% Part A: SVMs and Bayes Classifiers

%% 1. Support Vector Machines
%% Set up
close all;
clc;
clear;

load data_ps3_2.mat
C = 1000;

%% SVM: Problem 1
%% Data Set 1

% Testing with Klinear
svm_test(@Klinear, 1, C, set1_train, set1_test)
% TEST RESULTS: 0.0446 of test examples were misclassified.

% Testing with Kpoly
svm_test(@Kpoly, 2, C, set1_train, set1_test)
% TEST RESULTS: 0.0514 of test examples were misclassified.

% Testing with Kgaussian
svm_test(@Kgaussian, 1, C, set1_train, set1_test)
% TEST RESULTS: 0.0571 of test examples were misclassified.

% Conclusion:
% The best kernel is the linear one as the data set clearly shows a linear
% decision boundary separating the data. In these cases, it is best to 
% avoid overfitting, hence choosing the simplest model. The error rates on
% the test examples also support this claim as the linear decision boundary
% gave the lowest amounts of misclassified testing data (0.0446 of test 
% examples misclassified).

%% Data Set 2

% Testing with Klinear
svm_test(@Klinear, 1, C, set2_train, set2_test)
% TEST RESULTS: 0.273 of test examples were misclassified.

% Testing with Kpoly
svm_test(@Kpoly, 2, C, set2_train, set2_test)
% TEST RESULTS: 0.011 of test examples were misclassified.

% Testing with Kgaussian
svm_test(@Kgaussian, 1, C, set2_train, set2_test)
% TEST RESULTS: 0.014 of test examples were misclassified.

% Conclusion:
% The best kernel is the second order polynomial one as the data set
% clearly shows a parabola of degree 2 separating the data. In this case,
% the linear kernel would result in being underfitting, whereas the 
% Gaussian kernel would overfit. The error rates on the test examples also
% support this claim as the polynomial kernel of degree 2 gave the lowest
% amounts of misclassified testing data (0.011 of test examples were
% misclassified).

%% Data Set 3

% Testing with Klinear
svm_test(@Klinear, 1, C, set3_train, set3_test)
% TEST RESULTS: 0.471 of test examples were misclassified.

% Testing with Kpoly
svm_test(@Kpoly, 2, C, set3_train, set3_test)
% TEST RESULTS: 0.132 of test examples were misclassified.

% Testing with Kgaussian
svm_test(@Kgaussian, 1, C, set3_train, set3_test)
% TEST RESULTS: 0 of test examples were misclassified.

% Conclusion:
% The best kernel is the Gaussian one as the data set clearly shows groups
% of data points clustered, making it not separable with the linear or
% polynomial kernel. In this case, the Gaussian kernel is able to separate
% these groups, hence it is the best choice. The error rates on the test
% examples also support this claim as the Gaussian kernel produced no
% amounts of misclassified testing data.

%% SVM: Problem 2

% Testing with Klinear
svm_test_digital(@Klinear, 1, C, set4_train, set4_test)
% TEST RESULTS: 0.14 of test examples were misclassified.

% Testing with Kpoly
svm_test_digital(@Kpoly, 2, C, set4_train, set4_test)
% TEST RESULTS: 0.12 of test examples were misclassified.

% Testing with Kgaussian
svm_test_digital(@Kgaussian, 1.5, C, set4_train, set4_test)
% TEST RESULTS: 0.085 of test examples were misclassified.

% Conclusion:
% The linear kernel provided results of 0.14 misclassified test examples.
% The second order polynomial kernel provided results of 0.12 misclassified
% test examples.
% The Gaussian kernel provided results of 0.085 misclassified test 
% examples.



