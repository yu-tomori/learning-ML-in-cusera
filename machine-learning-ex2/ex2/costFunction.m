function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%
HofX = zeros(size(X, 1), 1);
eachCost = zeros(size(X, 1), 1);
simpleCost = zeros(size(X, 1), 1);

for i = 1:size(HofX, 1)
	HofX(i) = sigmoid(X(i, :) * theta);
	eachCost(i) = -y(i) * log(HofX(i)) - (1 - y(i)) * log(1 - HofX(i));
	simpleCost(i) = sigmoid(X(i, :) * theta) - y(i);
end
J = sum(eachCost) / size(eachCost, 1);

for j = 1:size(grad)
	sum = 0;
	for i = 1:size(X, 1)
		sum += simpleCost(i) * X(i, j);
	end
	grad(j) = sum / size(X, 1);
end









% =============================================================

end
