from scipy.optimize import minimize,check_grad
import numpy as np
import matplotlib.pyplot as plt

# Define the data points X and Y
X = np.array([-1, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
Y = np.array([-0.96, -0.577, -0.073, 0.377, 0.641, 0.66, 0.461, 0.134, -0.201, -0.434, -0.5, -0.393, -0.165, 0.099, 0.307, 0.396, 0.345, 0.182, -0.031, -0.219, -0.321])

# Functional approximation to fit
def func_to_fit(X, params):
    x0, x1, x2, x3, x4 = params
    return x0 + x1 * np.sin(x2 * X) + x3 * np.cos(x4 * X)

# Cost function (sum of squared errors)
def cost_function(params, X, Y):
    predictions = func_to_fit(X, params)
    return np.sum((predictions - Y) ** 2)

# Jacobian (gradient) of the cost function
def jacobian(params, X, Y):
    # Prediction from current parameters
    predictions = func_to_fit(X, params)
    # Derivatives of the fitting function with respect to each parameter
    J0 = -2 * (Y - predictions)  # Derivative w.r.t. x0
    J1 = -2 * (Y - predictions) * np.sin(params[2] * X)  # Derivative w.r.t. x1
    J2 = -2 * (Y - predictions) * params[1] * X * np.cos(params[2] * X)  # Derivative w.r.t. x2
    J3 = -2 * (Y - predictions) * np.cos(params[4] * X)  # Derivative w.r.t. x3
    J4 = 2 * (Y - predictions) * params[3] * X * np.sin(params[4] * X)  # Derivative w.r.t. x4
    # Stack to form the Jacobian
    return np.array([np.sum(J0), np.sum(J1), np.sum(J2), np.sum(J3), np.sum(J4)])

# Initial guess for parameters
initial_params = np.random.rand(5) +2

print(initial_params)


# Perform the optimization using 'BFGS' algorithm
result = minimize(cost_function, initial_params, args=(X, Y),jac=jacobian, method='Nelder-Mead', options={'disp': True, 'maxiter': 1000})

# Retrieve the optimal parameters
optimal_params = result.x
optimal_params

# Generate fitted function data points
X_fitted = np.linspace(min(X), max(X), 100)
Y_fitted = func_to_fit(X_fitted, optimal_params)

# Plot the fitted function and the original data
plt.figure(figsize=(10, 5))
plt.scatter(X, Y, color='red', label='Data points')
plt.plot(X_fitted, Y_fitted, label='Fitted function')
plt.title('Fitted Functional Approximation')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.grid(True)
plt.savefig('fitted_function3.pdf')

# Plot cost vs iteration
cost_history = result.hess_inv.diagonal()
plt.figure(figsize=(10, 5))
plt.plot(cost_history, label='Cost vs. Iteration')
plt.xlabel('Iteration number')
plt.ylabel('Cost')
plt.title('Cost vs. Iteration')
plt.legend()
plt.grid(True)
plt.savefig('cost_vs_iteration3.pdf')

print(optimal_params)
