function [lambda_vec, error_train, error_val] = ...
    validationCurve(X, y, Xval, yval)
%VALIDATIONCURVE Generate the train and validation errors needed to
%plot a validation curve that we can use to select lambda
%   [lambda_vec, error_train, error_val] = ...
%       VALIDATIONCURVE(X, y, Xval, yval) returns the train
%       and validation errors (in error_train, error_val)
%       for different values of lambda. You are given the training set (X,
%       y) and validation set (Xval, yval).
%
% Usage: [lambda_vec, error_train, error_val] = ...
%    validationCurve(X, y, Xval, yval);


% Selected values of lambda (you should not change this)
lambda_vec = [0 0.001 0.003 0.01 0.03 0.1 0.3 1 3 10]';

% You need to return these variables correctly.
error_train = zeros(length(lambda_vec), 1);
error_val = zeros(length(lambda_vec), 1);

for i = 1:length(lambda_vec)
	lambda = lambda_vec(i);
	[Theta1, Theta2, input_layer_size, hidden_layer_size, num_labels] ...
		= trainTitanicNN(X, y, lambda); % train the model
	error_train(i) = nnCostFunction([Theta1(:) ; Theta2(:)], ... 
		input_layer_size, hidden_layer_size, num_labels, ...
		X, y, 0);  % compute the cost for train set
	error_val(i) = nnCostFunction([Theta1(:) ; Theta2(:)], ... 
		input_layer_size, hidden_layer_size, num_labels, ...
		Xval, yval, 0); % compute the cost for validation set
end

close all;
plot(lambda_vec, error_train, lambda_vec, error_val);
legend('Train', 'Cross Validation');
xlabel('lambda');
ylabel('Error');

% =========================================================================

end
