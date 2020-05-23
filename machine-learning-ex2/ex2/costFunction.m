function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

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
	HofX(i) = sigmoid(-1 * X(i, :) * theta);
	eachCost = -y(i) * log(HofX(i)) - (1 - y(i)) * log(1 - HofX(i));
	simpleCost(i) = (sigmoid(-1 * X(i, :) * theta) - y(i));
end
J = round(sum(eachCost) / size(eachCost, 1), 3);
for i = 1:size(grad)
	grad(i) = round(simpleCost' * X(:, i) / size(simpleCost, 1), 4);
end









% =============================================================

end
