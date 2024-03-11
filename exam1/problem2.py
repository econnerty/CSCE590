import numpy as np
import matplotlib.pyplot as plt

# Given data points
t = np.array([-2, -1, 0, 1, 2])
y = np.array([2, -10, 0, 2, 1])

# Interpolate more t points between the given range
t_interpolated = np.linspace(t.min(), t.max(), 100)  # 100 points for a smooth curve

# Create the design matrix for the original t points
A = np.vstack([np.ones_like(t), t, t**2]).T

# Compute the pseudo-inverse of the design matrix
A_pinv = np.linalg.pinv(A)

# Calculate the weights (coefficients) for the quadratic polynomial
coefficients = A_pinv.dot(y)

# Evaluate the polynomial at the interpolated t points
y_interpolated = np.polyval(coefficients[::-1], t_interpolated)

# Plot the original data points and the interpolated polynomial
plt.scatter(t, y, label='Original data')
plt.plot(t_interpolated, y_interpolated, label='Interpolated polynomial', color='red')
plt.xlabel('t')
plt.ylabel('y(t)')
plt.title('Polynomial Interpolation')
plt.legend()
plt.savefig('problem2.pdf')