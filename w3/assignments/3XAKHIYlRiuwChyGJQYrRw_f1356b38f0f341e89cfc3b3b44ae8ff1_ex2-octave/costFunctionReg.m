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

%h = sigmoid(X*theta);

%t = [0; theta(2:size(theta), :)];
%J = 1/m * (-y'*log(h) - (1 - y)'*log(h)) + (lambda/(2*m))*(t'*t);

%grad = (1/m) * X' * (h - y);


% calculate cost function
h = sigmoid(X*theta);

% calculate penalty
% excluded the first theta value x0
t = [0 ; theta(2:size(theta), :)];
p = lambda*(t'*t)/(2*m);
J = ((-y)'*log(h) - (1-y)'*log(1-h))/m + p;

% calculate grads
grad = (X'*(h - y)+lambda*t)/m;



% =============================================================

end
