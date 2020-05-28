function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

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

m = size(X, 1);
n = size(theta, 1);
eachCost = zeros(m, 1);
simpleCost = zeros(m, 1);
grad = zeros(theta, 1);

for i = 1:m
    sigValue = sigmoid(X(i,:) * theta);
    eachCost(i) = -y(i) * log(sigValue) - (1 - y(i)) * log(1 - sigValue);
    simpleCost(i) = sigmoid(X(i, :) * theta) - y(i);
end

y = 0;
for i = 2:n
    y += theta(i) ^ 2;
end
J = sum(eachCost) / m + (lambda / (2 * m)) * y;

y = 0;
for i = 1:m
    y += simpleCost(i) * X(i, 1);
end
grad(1) = y / m;

for j = 2:n
    y = 0;
    for i = 1:m
        y += simpleCost(i) * X(i, j);
    end
    grad(j) = y / m + lambda * theta(j) / m;
end






% =============================================================

end
