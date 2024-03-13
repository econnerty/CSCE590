import numpy as np
import matplotlib.pyplot as plt

# Define the constraints and the objective function
def constraint1(x1, x2):
    return 4*x1**2 + 9*x2**4

def constraint2(x1, x2):
    return x1**2 + 3*x2**2

def objective_function(x1, x2):
    return (x1 - 3)**2 + (x2 - 3)**2

# Create a grid of x1 and x2 values
x1 = np.linspace(-1, 5, 400)
x2 = np.linspace(-2, 2, 400)

# Create a 2-D meshgrid of (x1, x2) values
X1, X2 = np.meshgrid(x1, x2)

# Compute constraint and objective function values for each (x1, x2) pair
C1 = constraint1(X1, X2)
C2 = constraint2(X1, X2)
F = objective_function(X1, X2)

# Initialize plot
fig, ax = plt.subplots(figsize=(8, 6))

# Constraint 1: 4x1^2 + 9x2^4 <= 36
contour1 = ax.contour(X1, X2, C1, levels=[36], colors='blue')
ax.clabel(contour1, inline=True, fontsize=8)

# Constraint 2: x1^2 + 3x2^2 = 3
contour2 = ax.contour(X1, X2, C2, levels=[3], colors='red')
ax.clabel(contour2, inline=True, fontsize=8)

# Feasible region: Highlight the region that satisfies both constraints
feasible_region = np.logical_and(C1 <= 36, C2 == 3)
ax.imshow(feasible_region, extent=(X1.min(), X1.max(), X2.min(), X2.max()), origin="lower", cmap="Greys", alpha=0.3)

# Objective function contours
contour_levels = np.linspace(0, F.max(), 50)
contourf = ax.contourf(X1, X2, F, levels=contour_levels, cmap="viridis", alpha=0.5)

# Plot the objective function contours
contour = ax.contour(X1, X2, F, levels=contour_levels, colors='black')
ax.clabel(contour, inline=True, fontsize=8)

# Labels and title
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_title('Nonlinear Programming Problem Feasible Region and Objective Function Contours')

# Show the plot
plt.colorbar(contourf, ax=ax, orientation="vertical")
plt.savefig('problem4.pdf')

