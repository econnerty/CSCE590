import numpy as np
import matplotlib.pyplot as plt
# Since S3 is the intersection of S1 and S2, we need to find the line where they intersect.
# For S1, x1 = x2 = x3, so we plug this into the equation of S2 (x1 + x2 + x3 = 1) to find the intersection.

# For S4, which is the Minkowski sum of S1 and S2, it is a bit more complex. We will use a simple example of points from S1 and S2 and add them.

# Define points for S1
x1_S1 = np.array([1, 1, 1])  # Example point where x1 = x2 = x3

# Define points for S2
x1_S2 = np.array([0.25, 0.25, 0.5])  # Example point where x1 + x2 + x3 = 1

# Minkowski sum of S1 and S2 (S4) for the example points
x1_S4 = x1_S1 + x1_S2

# Plot S3 and S4
fig = plt.figure(figsize=(16, 8))

# S3 is the intersection of S1 and S2, which is the line x1 = x2 = x3 where x1 + x2 + x3 = 1
# This line passes through the point (1/3, 1/3, 1/3)
ax3 = fig.add_subplot(121, projection='3d')
x1_S3 = np.linspace(-10, 10, 400)
x2_S3 = x1_S3
x3_S3 = x1_S3
ax3.plot(x1_S3, x2_S3, x3_S3, color='b', label='S3: Intersection of S1 and S2')
ax3.set_title('Set S3')
ax3.set_xlabel('x1')
ax3.set_ylabel('x2')
ax3.set_zlabel('x3')
ax3.legend()

# S4 is the Minkowski sum of S1 and S2
ax4 = fig.add_subplot(122, projection='3d')
# For simplicity, we just show the example point
ax4.scatter(x1_S4[0], x1_S4[1], x1_S4[2], color='m', label='S4: Minkowski sum of S1 and S2 (Example Point)')
ax4.set_title('Set S4')
ax4.set_xlabel('x1')
ax4.set_ylabel('x2')
ax4.set_zlabel('x3')
ax4.legend()

# Show the plots
plt.savefig('sets2.pdf')
