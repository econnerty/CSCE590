import numpy as np

# Define the matrix A and vector b
A = np.array([
    [1, 1, 1],
    [0, 0, 1],
    [1, 0, 1],
    [2,0,5],
    [-7, 8, 0],
    [1, 2, -1]
])

b = np.array([3, 1, 2,8, 0, 1])


# Least Squares Solution
def least_squares(A, b):
    AtA = np.dot(A.T, A)
    Atb = np.dot(A.T, b)
    x = np.linalg.inv(AtA).dot(Atb)
    return x

x_ls = least_squares(A, b)

# Regularized Least Squares Solution
def regularized_least_squares(A, b, lambda_value):
    AtA = np.dot(A.T, A)
    I = np.eye(A.shape[1]) * lambda_value
    Atb = np.dot(A.T, b)
    x = np.linalg.inv(AtA + I).dot(Atb)
    return x

lambda_value = 0.5
x_reg = regularized_least_squares(A, b, lambda_value)



#Steepest descent
def steepest_descent(A, b):
    # Initial guess (random or zeros)
    x = np.zeros(A.shape[1])

    # Learning rate
    alpha = 0.01

    # Maximum iterations
    max_iter = 100

    # Steepest Descent Optimization
    for i in range(max_iter):
        # Compute the residual
        r = A.dot(x) - b
        # Compute the gradient (A^T * residual)
        grad = A.T.dot(r)
        # Update the solution
        x = x - alpha * grad
        # Check if the solution satisfies the constraints
        if np.allclose(A.dot(x), b, atol=1e-5):
            break
    return x

x_linear = steepest_descent(A,b)#Solve with linear programming
print("Least Squares Solution:", x_ls)
print("Regularized Least Squares Solution:", x_reg)
print("Linear Programming:", x_linear)

from sklearn.decomposition import PCA



# Stack the solution vectors into a matrix where each row is a solution vector
solutions = np.array([x_ls, x_reg, x_linear])

# Initialize PCA and fit it to the solutions to reduce to 2 dimensions
pca = PCA(n_components=2)
solutions_2d = pca.fit_transform(solutions)

# The components tell you the contribution of each original feature in the new feature space
print("Principal axes in feature space:", pca.components_)
print("Explained variance by each principal component:", pca.explained_variance_ratio_)
print("Projected solutions onto 2D space:", solutions_2d)
print("Contribution of each original feature to the principal components:\n", pca.components_)
