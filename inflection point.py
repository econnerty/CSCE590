import numpy as np
import matplotlib.pyplot as plt

# Define the function and its derivatives to find the horizontal point of inflection
def f(x):
    return 3*x**4 - 4*x**3 + 1

def f_prime(x):
    return 12*x**3 - 12*x**2

def f_double_prime(x):
    return 36*x**2 - 24*x

# Create a range of x values
x = np.linspace(-2, 2, 400)
y = f(x)
y_prime = f_prime(x)
y_double_prime = f_double_prime(x)

# Find approximate point of inflection by finding where the second derivative is close to zero
# We take the absolute value since we are interested in the point where the second derivative changes sign
inflection_points = x[np.abs(y_double_prime) < 1e-2]

# Initialize the plot
plt.figure(figsize=(10, 6))

# Plot the function
plt.plot(x, y, label='Function $f(x) = 3x^4 - 4x^3 + 1$')

# Highlight the inflection points on the graph
for point in inflection_points:
    plt.plot(point, f(point), 'ro')  # 'ro' for red circle

# Plot the first and second derivatives
plt.plot(x, y_prime, label="First Derivative $f'(x)$")
plt.plot(x, y_double_prime, label="Second Derivative $f''(x)$")

# Add a legend
plt.legend()

# Add grid, title and labels
plt.grid(True)
plt.title('Function $f(x)$ with Horizontal Point of Inflection')
plt.xlabel('x')
plt.ylabel('f(x)')

# Show the plot
plt.savefig('plot2.pdf')

# Return the inflection point(s)
inflection_points, f(inflection_points)
