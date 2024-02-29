import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define the objective function
def objective(x1, x2):
    return (x1 - 3)**2 + (x2 - 3)**2

# Define the inequality constraint
def inequality(x1, x2):
    return 4*x1**2 + 9*x2**2 <= 36

# Define the equality constraint
def equality(x1):
    return (3 - x1**2) / 3

# Create a grid of points for x1 and x2
x1 = np.linspace(-10, 10, 100)
x2 = np.linspace(-10, 10, 100)
X1, X2 = np.meshgrid(x1, x2)

# Compute the objective function values
Z = objective(X1, X2)

# Apply the inequality constraint
ineq_mask = inequality(X1, X2) <= 36

# Apply the domain constraint
domain_mask = X1 >= -1

# Apply both the inequality and domain constraints to define the feasible region
feasible_region_mask = ineq_mask & domain_mask

# Create the plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the surface of the entire objective function with low alpha (transparency)
ax.plot_surface(X1, X2, Z, color='grey', alpha=0.1)

# Overlay the feasible region
X1_feasible = X1[feasible_region_mask]
X2_feasible = X2[feasible_region_mask]
Z_feasible = Z[feasible_region_mask]
ax.scatter(X1_feasible, X2_feasible, Z_feasible,alpha=.1, s=1, depthshade=False)  # Shading feasible region

# Plot the equality constraint as a curve
# Filter the x1 values to ensure x2 is real and within the feasible domain
x1_eq = x1[np.logical_and(x1 >= -1, x1**2 <= 3)]
x2_eq = equality(x1_eq)
Z_eq = objective(x1_eq, x2_eq)
ax.plot(x1_eq, x2_eq, Z_eq, color='red', linewidth=3, label='Equality Constraint')  # Curve for the equality constraint

# Set labels and title
ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('Objective Function Value')
ax.set_title('Feasible Region on the Objective Function Surface')

# Rotate the graph to get a better view
ax.view_init(elev=30, azim=60)  # Elevation and azimuthal angles

# Show the plot
plt.show()
