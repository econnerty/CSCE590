from sympy import symbols, diff, hessian, solve, Matrix, simplify, re, im

# Define the symbols
x1, x2, x3 = symbols('x1 x2 x3', real=True)

# Define the functions
f1 = x1**3 - 12*x1*x2 + 8*x2**3
f2 = (2*x1 - x2**2)**2 + (x2 - x3**2)**2 + (x3 - 1)**2

# Compute the gradients
grad_f1 = [diff(f1, var) for var in (x1, x2)]
grad_f2 = [diff(f2, var) for var in (x1, x2, x3)]

# Solve for critical points
critical_points_f1 = solve(grad_f1, (x1, x2), dict=True)  # Return solutions as dictionaries
critical_points_f2 = solve(grad_f2, (x1, x2, x3), dict=True)

# Compute the Hessians
hessian_f1 = hessian(f1, (x1, x2))
hessian_f2 = hessian(f2, (x1, x2, x3))

# Function to check the nature of critical points
def check_nature(hessian_matrix, critical_points):
    nature = []
    for point in critical_points:
        hessian_at_point = hessian_matrix.subs(point)
        eigenvalues = hessian_at_point.eigenvals()
        # Since eigenvalues are expressions, evaluate them numerically at the point if possible
        eval_eigenvalues = [val.evalf(subs=point) for val in eigenvalues]
        if all(val > 0 for val in eval_eigenvalues if val.is_real):
            nature.append('local minimum')
        elif all(val < 0 for val in eval_eigenvalues if val.is_real):
            nature.append('local maximum')
        else:
            nature.append('saddle point or undetermined')
    return nature

# Check the nature of the critical points for f1
nature_f1 = check_nature(hessian_f1, critical_points_f1)

# Check the nature of the critical points for f2
nature_f2 = check_nature(hessian_f2, critical_points_f2)

# Print the results
print("Critical points for f1:", critical_points_f1)
print("Nature of critical points for f1:", nature_f1)
print("Hessian Matrix for f1:", hessian_f1)

print("Critical points for f2:", critical_points_f2)
print("Nature of critical points for f2:", nature_f2)
print("Hessian Matrix for f2:", hessian_f2)
