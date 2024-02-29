from sympy import symbols, hessian, Function, Matrix

# Define the variables
x1, x2, x3 = symbols('x1 x2 x3')

# Define the function
f = 2*x1**2 + x1*x2 + x2**2 + x2*x3 + x3**2 - 6*x1 - 7*x2 - 8*x3 + 9

# Compute the gradient of f
gradient_f = Matrix([f.diff(x1), f.diff(x2), f.diff(x3)])

# Compute the Hessian matrix of f
hessian_f = hessian(f, (x1, x2, x3))

# Evaluate the gradient and Hessian at the point (0, 0, 0)
gradient_at_0 = gradient_f.subs({x1: 0, x2: 0, x3: 0})
hessian_at_0 = hessian_f.subs({x1: 0, x2: 0, x3: 0})

# Taylor series expansion around the point (0, 0, 0) up to the second order does not include quadratic terms
# The constant term f(0,0,0)
constant_term = f.subs({x1: 0, x2: 0, x3: 0})

# The linear term is gradient_f.T * (x - x_star)
x = Matrix([x1, x2, x3])
x_star = Matrix([0, 0, 0])
linear_term = gradient_f.T * (x - x_star)

# The quadratic term is 1/2 * (x - x_star).T * hessian_f * (x - x_star)
# However, since we are doing a second order approximation at the point (0,0,0), this term will be zero
# because (x - x_star) will be zero vector.

# Hence, the second order Taylor series approximation is just the constant term plus the linear term
taylor_approximation_at_0 = constant_term + linear_term[0]

gradient_at_0, hessian_at_0, taylor_approximation_at_0

print("The gradient of f with respect to x is:", gradient_at_0)
print("The Hessian matrix of f with respect to x is:", hessian_at_0)
print("The second order Taylor series approximation of f at (0, 0, 0) is:", taylor_approximation_at_0)
