from sympy import symbols, Matrix, solve

# Define the symbol
x = symbols('x')

# Define the matrix
M = Matrix([[x**4, x**3, x**2],
            [x**3, x**2, x],
            [x**2, x, 1]])

# Compute the eigenvalues
eigenvalues = M.eigenvals()

# Solve the characteristic polynomial for its roots (eigenvalues)
eigenvalues_solutions = solve(M.charpoly().as_expr())

eigenvalues, eigenvalues_solutions

print("The eigenvalues of the matrix are:", eigenvalues)
print("The solutions to the characteristic polynomial are:", eigenvalues_solutions)
