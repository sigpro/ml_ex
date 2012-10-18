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


    J = (-y'*log(sigmoid(X*theta)) - (1 - y)'*log(1-sigmoid(X*theta))) + (lambda/2)*sum(power(theta(2:end),2));
    J = J/m;
   %grad =  (sigmoid(X(2: length(theta)*theta(2: length(theta)))-y(2: length(theta))))'*X(2: length(theta))
   %      + lambda*theta(2: length(theta));
    grad(1) =  (sigmoid(X*theta)-y)'*X(:,1);
    grad(1) = grad(1)/m;
    grad(2:end) =  ((sigmoid(X*theta)-y)'*X(:,2:end))' + lambda*theta(2:end);
    grad(2:end) = grad(2:end)/m;



% =============================================================

end
