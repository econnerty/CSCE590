from sympy import symbols, diff, hessian, Matrix

# Define the symbols
x1, x2, x3 = symbols('x1 x2 x3')
c1, c2, c3, c4, c5, c6 = symbols('c1 c2 c3 c4 c5 c6')

# Define the function f(x)
f = c1*x1**2 + c2*x2**2 + c3*x3**2 + c4*x1*x2 + c5*x1*x3 + c6*x2*x3

# Derive its gradient
gradient_f = Matrix([diff(f, x1), diff(f, x2), diff(f, x3)])

# Compute the Hessian matrix
hessian_f = hessian(f, (x1, x2, x3))

# Output the function, gradient, and Hessian matrix
print("Function ", f)
print("Gradient " , gradient_f)
print("Hessian ",hessian_f)
