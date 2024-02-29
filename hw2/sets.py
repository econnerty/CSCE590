import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define the sets
def set_S1(x2, x3):
    return x2, x3

def set_S2(x1, x2):
    return x1, x2

# Create a meshgrid for the plot
x = np.linspace(-10, 10, 400)
X1, X2 = np.meshgrid(x, x)

# Set S1: x1 = x2, x2 = x3
X3_S1 = X2

# Set S2: x1 + x2 + x3 = 1, solving for x3 gives x3 = 1 - x1 - x2
X3_S2 = 1 - X1 - X2

# Create figure for S1 and S2
fig = plt.figure(figsize=(16, 8))

# Plot for S1
ax1 = fig.add_subplot(121, projection='3d')
ax1.plot_surface(X1, X2, X3_S1, alpha=0.5, rstride=100, cstride=100, color='r', edgecolor='none')
ax1.set_title('Set S1')
ax1.set_xlabel('x1')
ax1.set_ylabel('x2')
ax1.set_zlabel('x3')

# Plot for S2
ax2 = fig.add_subplot(122, projection='3d')
ax2.plot_surface(X1, X2, X3_S2, alpha=0.5, rstride=100, cstride=100, color='g', edgecolor='none')
ax2.set_title('Set S2')
ax2.set_xlabel('x1')
ax2.set_ylabel('x2')
ax2.set_zlabel('x3')

# Show the plots
plt.savefig('sets1.pdf')
