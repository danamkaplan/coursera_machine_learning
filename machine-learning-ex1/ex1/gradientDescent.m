function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %

    cost = 0;
    for i=1:m
      cost = cost + (theta(1)*X(i,1) + theta(2)*X(i,2) - y(i))*X(i, 1);
    end
    theta1_temp = theta(1) - alpha*cost/m;
    cost = 0;
    for i=1:m
      cost = cost + (theta(1)*X(i,1) + theta(2)*X(i,2) - y(i))*X(i, 2);
    end
    theta2_temp = theta(2) - alpha*cost/m;
    theta(1) = theta1_temp;
    theta(2) = theta2_temp;



    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
