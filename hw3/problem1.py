import numpy as np

# Define the function f
def f(x):
    return 5 * x[0]**2 + x[1]**2 + 4 * x[0] * x[1] - 14 * x[0] - 6 * x[1] + 20

# Define the gradient of f
def grad_f(x):
    return np.array([10 * x[0] + 4 * x[1] - 14, 2 * x[1] + 4 * x[0] - 6])

# Backtracking line search for step size
def line_search(x, grad, alpha=1.0, beta=0.8, c=0.0001):
    while f(x - alpha * grad) > f(x) - c * alpha * np.dot(grad, grad):
        alpha *= beta
    return alpha

# Starting point (we can choose an arbitrary point, e.g., (0,0))
starting_point = np.array([0.0, 0.0])

# Modified steepest descent algorithm to store iteration details
def steepest_descent_detailed(starting_point, tolerance=1e-5, max_iter=1000):
    x = starting_point
    iteration_details = []  # List to store iteration details

    for k in range(max_iter):
        grad = grad_f(x)
        grad_norm = np.linalg.norm(grad)
        if grad_norm < tolerance:
            break
        alpha = line_search(x, grad)
        iteration_details.append([k, x[0], x[1], -grad[0], -grad[1], grad_norm, alpha, f(x)])
        x = x - alpha * grad

    return iteration_details

# Execute the modified steepest descent algorithm
iteration_details = steepest_descent_detailed(starting_point)

# Extract the first 10 and last 10 iterations
first_10_iterations = iteration_details[:10]
last_10_iterations = iteration_details[-10:] if len(iteration_details) > 20 else iteration_details[-10:]

# Combine them for the table
table_iterations = first_10_iterations + last_10_iterations

# Format the iterations for display in a table format
import pandas as pd
table_columns = ['k', 'x^k_1', 'x^k_2', 'd^k_1', 'd^k_2', '||d^k||_2', 'alpha^k', 'f(x^k)']
df_iterations = pd.DataFrame(table_iterations, columns=table_columns)

df_iterations.to_csv('problem1.csv', index=False)
