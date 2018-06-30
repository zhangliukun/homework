function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));  %(2*1)

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

% 损失计算
hx = X * theta; 	%(12*1)
J = sum((hx - y).^2)/2/m + sum(theta(2:end).^2)*lambda/2/m;

% 梯度计算
grad(1) = X(:,1)'*(hx - y)/m;
grad(2:end) = X(:,2:end)'*(hx - y)/m + theta(2:end)*lambda/m;









% =========================================================================

grad = grad(:);

end
