function [] = svm_test_digital(kernel,param,C,train_data,test_data)

% Get the SVM
svm = svm_train(train_data, kernel, param, C);

% Verify for training data
y_est = sign(svm_discrim_func(train_data.X, svm));
error = find(y_est ~= train_data.y);

if (error)
    fprintf('WARNING: %d training examples were misclassified!!!\n',length(error));
end

% Evaluate against test data
y_est = sign(svm_discrim_func(test_data.X, svm));
error = find(y_est ~= test_data.y);

fprintf('TEST RESULTS: %g of test examples were misclassified.\n',...
    length(error)/length(test_data.y));