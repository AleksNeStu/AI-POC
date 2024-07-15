1) Gradient descent
   Gradient Descent is a popular optimization algorithm in machine learning and deep learning. It's used to minimize a function (like a loss function) by finding the value that gives the lowest function output.

Here's a simple explanation of how it works:

Initialize your parameters with some values. This could be a random initialization.

Calculate the gradient of the loss function at the current set of parameter values. The gradient is a vector that points in the direction of the greatest rate of increase of the function. However, since we want to minimize the function, we will go in the opposite direction of the gradient.

Update your parameters by a small step in the direction of the negative gradient. The size of the step is determined by the learning rate, a hyperparameter that you can set.

Repeat steps 2 and 3 until the gradient is close to zero, which means you've reached a minimum.

There are different variants of gradient descent, based on how much data we use to compute the gradient of the objective function. The three main forms are batch gradient descent, stochastic gradient descent, and mini-batch gradient descent.

Batch gradient descent: Uses the entire training dataset to compute the gradient of the cost function to the parameters.

Stochastic gradient descent (SGD): Uses only a single data point or example to compute the gradient at each step.

Mini-batch gradient descent: A compromise between batch gradient descent and SGD. It uses a mini-batch of 'n' training examples to compute the gradient at each step. This is the most common type of gradient descent within deep learning.

The key idea is to minimize the cost/loss/objective function to reach the optimal values where the model has the least error.


2) 