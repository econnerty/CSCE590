from sympy import symbols, Matrix, Function, diff

# Define symbols for dimensions
m, n = 3, 3  # Example dimensions

# Define the symbol for vector x (domain of g) and vector y (domain of f)
x = Matrix(m, 1, symbols('x1:%d' % (m+1)))
y = Matrix(n, 1, symbols('y1:%d' % (n+1)))

# Define the matrix A (m x n matrix)
A = Matrix(m, n, symbols('a1:%d:%d' % (m*n+1, n+1)))

# Define the function g(x) which takes an m-dimensional vector and returns a scalar
g = Function('g')(*x)

# Define the function f(y) in terms of g and A*y
f = g.subs(list(zip(x, A*y)))

# Compute the gradient of f with respect to y
grad_f = Matrix([diff(f, yi) for yi in y])

# Compute the Hessian matrix of f with respect to y
hessian_f = Matrix([[diff(grad_f[j], y[i]) for i in range(n)] for j in range(n)])

# Display the gradient and Hessian matrix
grad_f, hessian_f
print("The gradient of f with respect to y is:", grad_f)
print("The Hessian matrix of f with respect to y is:", hessian_f)
