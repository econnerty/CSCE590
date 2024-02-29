

# Import numpy for numerical computations
import itertools
import numpy as np

# Create a grid of points within the domain
x_vals = np.linspace(0, 2 * np.pi, 400)
y_vals = np.linspace(0, 2 * np.pi, 400)
X, Y = np.meshgrid(x_vals, y_vals)

# Evaluate the function on the grid
Z = np.sin(X) + np.sin(Y) + np.sin(X + Y)

# Find local minima and maxima by comparing each point with its neighbors
def find_local_extrema(Z):
    local_minima = []
    local_maxima = []

    for i in range(1, Z.shape[0] - 1):
        for j in range(1, Z.shape[1] - 1):
            region = Z[i-1:i+2, j-1:j+2]
            center = Z[i, j]
            if center == np.min(region) and center < Z[i-1, j] and center < Z[i+1, j] and center < Z[i, j-1] and center < Z[i, j+1]:
                local_minima.append((X[i, j], Y[i, j]))
            elif center == np.max(region) and center > Z[i-1, j] and center > Z[i+1, j] and center > Z[i, j-1] and center > Z[i, j+1]:
                local_maxima.append((X[i, j], Y[i, j]))

    return local_minima, local_maxima

local_minima, local_maxima = find_local_extrema(Z)

# Because the grid might identify multiple points very close to each other as minima/maxima,
# we will cluster these points and consider each cluster a single extrema point.

from scipy.cluster.vq import kmeans, vq

# The previous error was caused by attempting to create more clusters than there are points.
# To fix this, we will ensure the number of clusters does not exceed the number of points.

# Check if there are enough points for clustering and perform k-means if so
def cluster_extrema(extrema_points, num_clusters=3):
    if not extrema_points:
        return []
    
    # Ensure we don't exceed the number of extrema points with the number of clusters
    num_clusters = min(num_clusters, len(extrema_points))
    
    # Perform k-means clustering
    centroids, _ = kmeans(extrema_points, num_clusters)
    return centroids.tolist()

# Apply the clustering to the local minima and maxima found
unique_local_minima = cluster_extrema(local_minima)
unique_local_maxima = cluster_extrema(local_maxima)

unique_local_minima, unique_local_maxima

