function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m

y_matrix = eye(num_labels)(y,:); 
% a1 equals the X input matrix with a column of 1's added (bias units) as the first column.
a1 = [ones([m,1]) X];
% z2 equals the product of a1 and Θ1
z2 = a1*Theta1';
% a2 is the result of passing z2 through g()
% Then add a column of bias units to a2 (as the first column).
a2 = sigmoid(z2);
a2 = [ones([m,1]) a2];
% z3 equals the product of a2 and Θ2
z3 = a2*Theta2';
% a3 is the result of passing z3 through g()
a3 = sigmoid(z3);

% 3 - Compute the unregularized cost according to ex4.pdf (top of Page 5), 
% using a3, your y_matrix, and m (the number of training examples). 
% Note that the 'h' argument inside the log() function is exactly a3. 
% Cost should be a scalar value. Since y_matrix and a3 are both matrices, 
% you need to compute the double-sum.
%
% Remember to use element-wise multiplication with the log() function. Also, we're using the natural log, not log10().

inner_cost = (-y_matrix).*log(a3)-(1-y_matrix).*log(1-a3);
J = (sum(sum(inner_cost, 2)))/m;

% Now you can run ex4.m to check the unregularized cost is correct, then you can submit this portion to the grader.

% Adding regularized term to cost
Theta1_reg = sum(sum(Theta1(:,2:end).^2, 2));
Theta2_reg = sum(sum(Theta2(:,2:end).^2, 2));
reg =(lambda/(2*m))*(Theta1_reg + Theta2_reg);
J = J + reg;

%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%



















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
