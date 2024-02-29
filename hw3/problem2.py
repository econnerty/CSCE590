import numpy as np
import pandas as pd

# Define the quadratic function f
def quadratic_f(x, Q, C):
    return 0.5 * x.T @ Q @ x - C.T @ x + 10

# Define the gradient of the quadratic function f
def grad_quadratic_f(x, Q, C):
    return Q @ x - C

# Backtracking line search for step size
def line_search(x,Q,C,grad, alpha=1.0, beta=0.8, c=0.0001):
    while quadratic_f(x - alpha * grad,Q,C) > quadratic_f(x,Q,C) - c * alpha * np.dot(grad, grad):
        alpha *= beta
    return alpha

# Starting point
x0 = np.array([40, -100])

# Parameters for the first case
Q1 = np.array([[20, 5], [5, 2]])
C1 = np.array([14, 6])

# Parameters for the second case
Q2 = np.array([[20, 5], [5, 16]])
C2 = np.array([14, 6])

# Steepest descent algorithm (reusing previous implementation)
def steepest_descent_quadratic(starting_point, Q, C, tolerance=10e-6, max_iter=1000):
    x = starting_point
    iteration_details = []  # List to store iteration details

    for k in range(max_iter):
        grad = grad_quadratic_f(x, Q, C)
        grad_norm = np.linalg.norm(grad)
        if grad_norm < tolerance:
            break
        alpha = line_search(x,Q,C, grad)
        iteration_details.append([k, x[0], x[1], -grad[0], -grad[1], grad_norm, alpha, quadratic_f(x, Q, C)])
        x = x - alpha * grad

    return iteration_details

# Run steepest descent for both cases
iteration_details_case1 = steepest_descent_quadratic(x0, Q1, C1)
iteration_details_case2 = steepest_descent_quadratic(x0, Q2, C2)

# We will only show the first 10 and last 10 iterations for each case
def extract_iterations(iteration_details):
    first_10 = iteration_details[:10]
    last_10 = iteration_details[-10:] if len(iteration_details) > 20 else iteration_details[-10:]
    return first_10 + last_10

table_case1 = extract_iterations(iteration_details_case1)
table_case2 = extract_iterations(iteration_details_case2)


table_columns = ['k', 'x^k_1', 'x^k_2', 'd^k_1', 'd^k_2', '||d^k||_2', 'alpha^k', 'f(x^k)']
# Convert to pandas DataFrame for display
df_case1 = pd.DataFrame(table_case1, columns=table_columns)
df_case2 = pd.DataFrame(table_case2, columns=table_columns)

# Save the results to a CSV file
df_case1.to_csv('problem2_case1.csv', index=False)
df_case2.to_csv('problem2_case2.csv', index=False)

(df_case1, df_case2)
