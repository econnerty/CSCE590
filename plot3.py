import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Define the function
def f1(x):
    return x + np.sin(x)

# Define the range for x and calculate f(x)
x_range = np.linspace(-2*np.pi, 2*np.pi, 1000)
y_range = f1(x_range)

# Find the local minimizer and maximizer within the range of -2pi to 2pi
minimizer_local = minimize(f1, 0, bounds=[(-2*np.pi, 2*np.pi)])
maximizer_local = minimize(lambda x: -f1(x), 3, bounds=[(-2*np.pi, 2*np.pi)])

# Plot the function along with the local minimizer and maximizer
plt.figure(figsize=(12, 6))
plt.plot(x_range, y_range, label='$f(x) = x + \sin(x)$')
#plt.scatter(minimizer_local.x, minimizer_local.fun, color='red', label='Local Minimizer', zorder=5)
#plt.scatter(maximizer_local.x, -maximizer_local.fun, color='green', label='Local Maximizer', zorder=5)
plt.axhline(0, color='black', linewidth=0.5, linestyle='--')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Function $f(x) = x + \sin(x)')
plt.legend()
plt.grid(True)
plt.savefig('sin.pdf')
