import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define the functions
def func1(x1, x2):
    return x1**2 + x2**2

def func2(x1, x2):
    return -x1 * np.log(x1) - x2 * np.log(x2)

def func3(x1, x2):
    return np.abs(x1) + np.abs(x2)

# Create a grid of x1 and x2 values
x = np.linspace(0.1, 2, 400)  # Start at 0.1 to avoid log(0)
y = np.linspace(0.1, 2, 400)
X, Y = np.meshgrid(x, y)

# Compute the Z values for each function
Z1 = func1(X, Y)
Z2 = func2(X, Y)
Z3 = func3(X, Y)

# Create figure for the plots
fig = plt.figure(figsize=(18, 6))

# Plot for func1
ax1 = fig.add_subplot(131, projection='3d')
ax1.plot_surface(X, Y, Z1, cmap='viridis')
ax1.set_title('Function 1: $f(x) = x_1^2 + x_2^2$')

# Plot for func2
ax2 = fig.add_subplot(132, projection='3d')
ax2.plot_surface(X, Y, Z2, cmap='viridis')
ax2.set_title('Function 2: $f(x) = -x_1 \ln(x_1) - x_2 \ln(x_2)$')

# Plot for func3
ax3 = fig.add_subplot(133, projection='3d')
ax3.plot_surface(X, Y, Z3, cmap='viridis')
ax3.set_title('Function 3: $f(x) = |x_1| + |x_2|$')

# Show the plots
plt.show()

# Save the figure to a PDF file
fig.savefig('functions_plot.pdf')
