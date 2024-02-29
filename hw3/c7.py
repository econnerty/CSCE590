from sympy import symbols, sqrt, diff, Matrix
# Define the symbols for x1 and x2
x1, x2 = symbols('x1 x2')
# Define the function
f = 100 * (x2 - x1**2)**2 + (1 - x1)**2

# Compute the gradient of f
gradient_f = Matrix([diff(f, var) for var in (x1, x2)])

# Evaluate the gradient at the point (0, 0)
gradient_at_0 = gradient_f.subs({x1: 0, x2: 0})

# The descent direction is given by the negative of the gradient.
# For a descent direction, we are interested in the direction in which the function decreases,
# which is opposite to the direction of the gradient.
descent_direction = -gradient_at_0

gradient_at_0, descent_direction
