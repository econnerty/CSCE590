import matplotlib.pyplot as plt
import numpy as np

# Create a grid of values
x1 = np.linspace(0, 10, 400)
x2 = np.linspace(0, 10, 400)

X1, X2 = np.meshgrid(x1, x2)

# Adjust the code provided by the user to correctly plot the feasible region, taking into account all constraints.

# Define the constraints in terms of x1 and x2
def constraint1(x2):
    return 4 - (8/3) * x2  # x1 + (8/3)*x2 <= 4

def constraint2(x2):
    return 2 - x2  # x1 + x2 <= 2

# Since 2*x1 <= 3 simplifies to x1 <= 1.5, we define it as a constant
constraint3_x1 = 1.5

# Create a grid of x2 values
x2_values = np.linspace(0, 10, 400)

# Calculate the constraint lines
c1_values = constraint1(x2_values)
c2_values = constraint2(x2_values)

# Initialize the plot
plt.figure(figsize=(10, 10))

# Plot the constraint lines
plt.plot(c1_values, x2_values, label=r'$x1 + \frac{8}{3}x2 \leq 4$')
plt.plot(c2_values, x2_values, label=r'$x1 + x2 \leq 2$')
plt.axvline(x=constraint3_x1, color='r', linestyle='-', label=r'$2x1 \leq 3$')

# Shade the feasible region considering all three constraints
# First, find the minimum of the two constraint functions at each point along x2
x1_feasible = np.minimum(c1_values, c2_values)

# The feasible region for x1 is also limited by the vertical line x1 <= 1.5 (constraint3_x1)
x1_feasible = np.minimum(x1_feasible, np.full_like(x1_feasible, constraint3_x1))

# The feasible region for x2 is limited by x2 <= 2 (from constraint2 when x1=0)
x2_feasible_limit = 2

# Shade the region where x1_feasible is greater than or equal to 0 and x2 is less than or equal to x2_feasible_limit
plt.fill_betweenx(x2_values, 0, x1_feasible, where=(x2_values <= x2_feasible_limit), color='yellow', alpha=0.5)

# Set the limits of the plot to the area of interest
plt.xlim(0, 3)
plt.ylim(0, 3)

# Add labels and title
plt.xlabel(r'$x1$')
plt.ylabel(r'$x2$')
plt.title('Feasible Region for the Given Constraints')

# Add the legend
plt.legend()

# Show the plot
plt.savefig('feasible_region.pdf')
