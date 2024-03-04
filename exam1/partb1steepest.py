import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import line_search

# Define the objective function
def objective_function(x):
    return 5*x[0]**2 + x[1]**2 + 2*x[2]**2 + 4*x[0]*x[1] - 14*x[0] - 6*x[1] + 20

# Define the gradient of the objective function
def gradient(x):
    return np.array([10*x[0] + 4*x[1] - 14, 2*x[1] + 4*x[0] - 6, 4*x[2]])

# Bisection method for step size determination
def bisection_step_size(x, grad, objective_function, gradient, low, high, tol=1e-5, max_iterations=100):
    c = 1e-4  # Armijo condition constant
    for _ in range(max_iterations):
        mid = (low + high) / 2
        new_x = x - mid * grad
        if objective_function(new_x) < objective_function(x) - c * mid * np.dot(grad, grad) and np.all(np.isfinite(new_x)):
            high = mid
        else:
            low = mid
        if high - low < tol:
            break
    return mid

# Steepest Descent implementation with dynamic step size
def steepest_descent(initial_point, step_size_rule='constant', initial_step_size=1.0, max_iterations=100):
    x = np.array(initial_point)
    cost_history = [objective_function(x)]
    alpha = initial_step_size  # Initial step size for 'constant' and 'diminishing'

    for iteration in range(max_iterations):
        grad = gradient(x)
        
        if step_size_rule == 'constant':
            pass  # alpha remains constant
        elif step_size_rule == 'minimization':
            # Line search (backtracking)
            alpha = line_search(objective_function, gradient, x, -grad)[0]
            if alpha is None: alpha = initial_step_size  # Fallback if line search fails
        elif step_size_rule == 'diminishing':
            alpha = initial_step_size / (1 + iteration)  # Diminishing step size
        elif step_size_rule == 'bisection':
            alpha = bisection_step_size(x, grad, objective_function, gradient, low=1e-5, high=alpha)
        else:
            raise ValueError("Unknown step size rule.")
        
        # Update x
        x = x - alpha * grad
        cost = objective_function(x)
        cost_history.append(cost)
        
        # Stop if the change in the cost function is below a threshold or if non-finite value is encountered
        if np.abs(cost_history[-2] - cost) < 1e-9 or not np.isfinite(cost):
            break

    return x, cost_history

# Initial guess and initial step size
initial_point = [0, 0, 0]
initial_step_size = 0.001  # Reduced initial step size

# Apply Steepest Descent with different step size rules
minimizer_constant, cost_history_constant = steepest_descent(initial_point, 'constant', initial_step_size)
minimizer_minimization, cost_history_minimization = steepest_descent(initial_point, 'minimization', initial_step_size)
minimizer_diminishing, cost_history_diminishing = steepest_descent(initial_point, 'diminishing', initial_step_size)
minimizer_bisection, cost_history_bisection = steepest_descent(initial_point, 'bisection', initial_step_size)

# Plot the results
plt.figure(figsize=(12, 8))
plt.plot(cost_history_constant, label='Constant Step Size', color='red')
plt.plot(cost_history_minimization, label='Minimization Rule', color='blue')
plt.plot(cost_history_diminishing, label='Diminishing Step Size', color='green')
plt.plot(cost_history_bisection, label='Bisection Rule', color='purple')
plt.xlabel('Iteration')
plt.ylabel('Cost')
plt.title('Cost vs Iteration (Steepest Descent with Different Step Sizes)')
plt.legend()
plt.savefig('steepest_descent.pdf')
