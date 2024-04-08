import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv

# Data points
t = np.array([-2, -1, 0, 1, 2])
y = np.array([2, -10, 0, 2, 1])

# Polynomial fitting function
def polynomial(t, coeffs):
    """Evaluate a polynomial at times t with given coefficients."""
    return np.polyval(coeffs[::-1], t)

# Error function
def error_function(coeffs, t, y):
    """Calculate the sum of the squares of the differences."""
    p = polynomial(t, coeffs)
    return np.sum((p - y) ** 2)

# Gradient of the error function
def gradient(coeffs, t, y):
    """Calculate the gradient of the error function."""
    p = polynomial(t, coeffs)
    grad = np.array([2 * np.sum((p - y) * t**i) for i in range(len(coeffs))])
    return grad

# Hessian matrix of the error function
def hessian(coeffs, t):
    """Calculate the Hessian matrix of the error function."""
    H = np.array([[2 * np.sum(t**(i+j)) for i in range(len(coeffs))] for j in range(len(coeffs))])
    return H

# Steepest descent algorithm
def steepest_descent(t, y, coeffs, learning_rate, max_iter, tol):
    error_history = []
    for _ in range(max_iter):
        grad = gradient(coeffs, t, y)
        coeffs -= learning_rate * grad
        error = error_function(coeffs, t, y)
        error_history.append(error)
        if np.linalg.norm(grad) < tol:
            break
    return coeffs, error_history

# Newton's method
def newton_method(t, y, coeffs, max_iter, tol):
    error_history = []
    for _ in range(max_iter):
        grad = gradient(coeffs, t, y)
        H = hessian(coeffs, t)
        coeffs -= np.dot(inv(H), grad)
        error = error_function(coeffs, t, y)
        error_history.append(error)
        if np.linalg.norm(grad) < tol:
            break
    return coeffs, error_history

# Initial coefficients for quadratic polynomial (guess)
initial_coeffs = np.array([0.0, 0.0, 0.0])

# Parameters for the optimization
learning_rate = 0.01
max_iter = 100
tol = 1e-6

# Fit the polynomial using steepest descent
coeffs_sd, error_history_sd = steepest_descent(t, y, initial_coeffs.copy(), learning_rate, max_iter, tol)

# Fit the polynomial using Newton's method
coeffs_nm, error_history_nm = newton_method(t, y, initial_coeffs.copy(), max_iter, tol)

# Generate fitted polynomial data points
t_fitted = np.linspace(min(t), max(t), 100)
y_fitted_sd = polynomial(t_fitted, coeffs_sd)
y_fitted_nm = polynomial(t_fitted, coeffs_nm)

# Plot the fitted polynomial and the original data
plt.figure(figsize=(14, 6))

# Plot for steepest descent
plt.subplot(1, 2, 1)
plt.scatter(t, y, color='red', label='Data points')
plt.plot(t_fitted, y_fitted_sd, label='Steepest Descent Fit')
plt.title('Steepest Descent Polynomial Fit')
plt.xlabel('t')
plt.ylabel('y')
plt.legend()

# Plot for Newton's method
plt.subplot(1, 2, 2)
plt.scatter(t, y, color='red', label='Data points')
plt.plot(t_fitted, y_fitted_nm, label='Newton\'s Method Fit')
plt.title('Newton\'s Method Polynomial Fit')
plt.xlabel('t')
plt.ylabel('y')
plt.legend()

# Show the plot
plt.tight_layout()
plt.show()

# Output coefficients and error histories for further use
(coeffs_sd, coeffs_nm, error_history_sd, error_history_nm)
