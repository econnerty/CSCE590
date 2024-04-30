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

def gram_schmidt(A):
    m, n = A.shape
    Q = np.zeros((m, n))
    R = np.zeros((n, n))
    
    for k in range(n):
        Q[:, k] = A[:, k]
        for i in range(k):
            R[i, k] = np.dot(Q[:, i].T, A[:, k])
            Q[:, k] -= R[i, k] * Q[:, i]
        R[k, k] = np.linalg.norm(Q[:, k])
        Q[:, k] /= R[k, k]
    
    return Q, R

def back_substitution(R, Qb):
    n = R.shape[1]
    x = np.zeros(n)
    for i in reversed(range(n)):
        x[i] = Qb[i]
        for j in range(i+1, n):
            x[i] -= R[i, j] * x[j]
        x[i] /= R[i, i]
    return x

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

# Linear Programming Solver (Relaxed to Least Squares for simplicity)
# Here we simply use the least squares as a placeholder
# In practice, for LP, you would use a simplex method or similar
# QR Decomposition using Gram-Schmidt
Q, R = gram_schmidt(A)

# Compute Q^T * b
Qb = np.dot(Q.T, b)

# Solve the system using back substitution
x_qr = back_substitution(R[:A.shape[1], :], Qb[:A.shape[1]])

print("Least Squares Solution:", x_ls)
print("Regularized Least Squares Solution:", x_reg)
print("QR Decomposition:", x_qr)

from sklearn.decomposition import PCA



# Stack the solution vectors into a matrix where each row is a solution vector
solutions = np.array([x_ls, x_reg, x_qr])

# Initialize PCA and fit it to the solutions to reduce to 2 dimensions
pca = PCA(n_components=2)
solutions_2d = pca.fit_transform(solutions)

# The components tell you the contribution of each original feature in the new feature space
print("Principal axes in feature space:", pca.components_)
print("Explained variance by each principal component:", pca.explained_variance_ratio_)
print("Projected solutions onto 2D space:", solutions_2d)
print("Contribution of each original feature to the principal components:\n", pca.components_)
