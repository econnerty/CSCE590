import numpy as np
import matplotlib.pyplot as plt

# Define the cost function J(x)
def cost_function(A, x, b):
    return np.sum((A @ x - b)**2)

# Define the gradient of J(x)
def gradient(A, x, b):
    return 2 * A.T @ (A @ x - b)

# Define the gradient descent algorithm
def gradient_descent(A, b, initial_x, learning_rate, max_iterations):
    x = initial_x
    cost_history = []
    for _ in range(max_iterations):
        grad = gradient(A, x, b)
        x = x - learning_rate * grad
        cost = cost_function(A, x, b)
        cost_history.append(cost)
    return x, cost_history

# Given data points
X = np.array([-1, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
Y = np.array([-0.96, -0.577, -0.073, 0.377, 0.641, 0.66, 0.461, 0.134, -0.201, -0.434, -0.5, -0.393, -0.165, 0.099, 0.307, 0.396, 0.345, 0.182, -0.031, -0.219, -0.321])

# Construct the design matrix A
A = np.column_stack((np.ones(X.shape), np.sin(X) + np.cos(X), np.sin(2*X) + np.cos(2*X), np.sin(3*X) + np.cos(3*X)))

# Initial guess for the coefficients x
initial_x = np.zeros(4)

# Set the learning rate and max_iterations for the gradient descent
learning_rate = 0.01
max_iterations = 4000

# Solve for the coefficients using gradient descent
final_x, cost_history = gradient_descent(A, Y, initial_x, learning_rate, max_iterations)

# Plot the cost history
plt.plot(cost_history)
plt.xlabel('Iteration')
plt.ylabel('Cost')
plt.title('Cost vs Iteration')
plt.savefig('problem3_cost.pdf')

# Return the final coefficients
print(final_x)

# Define the function as per the given problem
def f(X, x0, x1, x2, x3):
    return x0 + x1 * (np.sin(X) + np.cos(X)) + x2 * (np.sin(2*X) + np.cos(2*X)) + x3 * (np.sin(3*X) + np.cos(3*X))

# Given coefficients (replace with your final_x from the optimization)
coefficients = final_x 
# Interpolated points
X_interp = np.linspace(min(X), max(X), 1000)  # Assuming X and Y are defined as in the problem
Y_interp = f(X_interp, *coefficients)

# Plot the predicted function and the actual data points
plt.figure(figsize=(12, 6))
plt.plot(X_interp, Y_interp, label='Predicted Function')
plt.scatter(X, Y, color='red', label='Actual Data')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.title('Predicted vs Actual Function')
plt.savefig('problem3_predicted.pdf')

