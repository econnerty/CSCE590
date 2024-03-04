import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# Define the objective function
def objective_function(x):
    return 5 * x[0]**2 + x[1]**2 + 2 * x[2]**2 + 4 * x[0] * x[1] - 14 * x[0] - 6 * x[1] + 20

# To track the cost at each iteration, we will use a callback function
cost_history = []

def callback(x):
    cost = objective_function(x)
    cost_history.append(cost)

# Initial guess
x0 = np.array([0, 0, 0])

# Call the minimization using the Nelder-Mead algorithm
result = minimize(objective_function, x0, method='Nelder-Mead', callback=callback)

# Plot cost vs iteration
plt.plot(cost_history, label='Numerical Calculations')
plt.xlabel('Iteration')
plt.ylabel('Cost')
plt.title('Cost vs Iteration')
plt.legend()
plt.savefig('numerical.pdf')

# Return the result of the optimization and the final value of the objective function
result.x, result.fun
