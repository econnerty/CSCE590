from scipy.optimize import minimize
import numpy as np
import matplotlib.pyplot as plt


def f2(x):
    return (2*x[0] - x[1])**2 + (x[1] - x[2])**2 + (x[2] - 1)**2


# For f2, since it's a multivariate function, we'll need to provide an initial guess for each variable
initial_guess = [0, 0, 0]
minimizer_f2 = minimize(f2, initial_guess)


# Since f2 is a multivariate function, we'll plot it in 3D
from mpl_toolkits.mplot3d import Axes3D
X1, X2, X3 = np.meshgrid(np.linspace(-2, 2, 30), np.linspace(-2, 2, 30), np.linspace(-2, 2, 30))
Y = f2([X1, X2, X3])

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(minimizer_f2.x[0], minimizer_f2.x[1], minimizer_f2.fun, color='red', label='Global Minimizer', s=100)
ax.plot_surface(X1[:, :, 0], X2[:, :, 0], Y[:, :, 0], cmap='viridis', alpha=0.6)
ax.set_title('Plot of $f(x) = (2x_1 - x_2)^2 + (x_2 - x_3)^2 + (x_3 - 1)^2$ with Global Minimizer')
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_zlabel('$f(x_1, x_2, x_3)$')
ax.legend()
plt.savefig('globalmin.pdf')

# Output the minimizers and maximizers for
print(minimizer_f2)
