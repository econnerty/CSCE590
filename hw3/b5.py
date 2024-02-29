from sympy import symbols, sqrt

# Define the symbols for x1 and x2
x1, x2 = symbols('x1 x2')

# Define the constraints
constraint1 = 1 - x1**2 - x2**2
constraint2 = sqrt(2) - x1 - x2
constraint3 = x2

# Define the points
points = {
    'x_a': (1/2, 1/2),
    'x_b': (1, 0),
    'x_c': (-1, 0),
    'x_d': (-1/2, 0),
    'x_e': (1/sqrt(2), 1/sqrt(2))
}

# Function to check the status of the point with respect to the feasible region
def check_point(point):
    x1_val, x2_val = point
    status = {
        'constraint1': constraint1.subs({x1: x1_val, x2: x2_val}),
        'constraint2': constraint2.subs({x1: x1_val, x2: x2_val}),
        'constraint3': constraint3.subs({x1: x1_val, x2: x2_val})
    }
    return status

# Check each point and store the results
results = {point_label: check_point(point) for point_label, point in points.items()}
results
