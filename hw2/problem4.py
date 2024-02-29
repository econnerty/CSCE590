from sympy import symbols, diff, exp, hessian, solve, re,im

# Define the symbols
x, y, z = symbols('x y z', real=True)

# Define the functions
f1 = exp(x**2 + y**2 + z**2) - x**4 - y**6 - z**6
f2 = x**3 + exp(3*y) - 3*x*exp(y)

# Compute the gradients
grad_f1 = [diff(f1, var) for var in (x, y, z)]
grad_f2 = [diff(f2, var) for var in (x, y)]

# Solve for critical points
critical_points_f1 = solve(grad_f1, (x, y, z), dict=True)
critical_points_f2 = solve(grad_f2, (x, y), dict=True)

# Compute the Hessians
hessian_f1 = hessian(f1, (x, y, z))
hessian_f2 = hessian(f2, (x, y))

# Function to evaluate the definiteness of a matrix at given points
def eval_definiteness(hessian_matrix, points):
    for point in points:
        # Substitute the point into the Hessian matrix
        hessian_at_point = hessian_matrix.subs(point)
        # Calculate the eigenvalues of the Hessian matrix
        eigenvalues = hessian_at_point.eigenvals()
        # Filter out complex eigenvalues
        real_eigenvalues = [val for val in eigenvalues if im(val) == 0]

        if not real_eigenvalues:
            print(f"Critical point {point} has complex eigenvalues, cannot determine nature.")
            continue
        
        # Check the signs of the real eigenvalues
        if all(val > 0 for val in real_eigenvalues):
            print(f"Critical point {point} is a local minimum.")
        elif all(val < 0 for val in real_eigenvalues):
            print(f"Critical point {point} is a local maximum.")
        else:
            print(f"Critical point {point} is a saddle point or undetermined.")

# Evaluate definiteness for f1 at its critical points
print("Analyzing f1...")
eval_definiteness(hessian_f1, critical_points_f1)

# Evaluate definiteness for f2 at its critical points
print("\nAnalyzing f2...")
eval_definiteness(hessian_f2, critical_points_f2)
