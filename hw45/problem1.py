import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.optimize import minimize_scalar

# Define the objective function
def f(x):
    return 5*x[0]**2 + x[1]**2 + 2*x[2]**2 + 4*x[0]*x[1] - 14*x[0] - 6*x[1] + 20

# Define the gradient of the function
def grad_f(x):
    return np.array([
        10*x[0] + 4*x[1] - 14,
        2*x[1] + 4*x[0] - 6,
        4*x[2]
    ])

# Define the Hessian of the function
def hess_f(x):
    return np.array([
        [10, 4, 0],
        [4, 2, 0],
        [0, 0, 4]
    ])

# Steepest Descent (SD) with Bisection for stepsize selection
def steepest_descent_bisection(f, grad_f, x0, tol=1e-6, max_iter=1000):
    x = x0
    it = 0
    values = []
    x_hist = [x0]
    
    while np.linalg.norm(grad_f(x)) > tol and it < max_iter:
        it += 1
        direction = -grad_f(x)
        
        # Define the function for stepsize selection via Bisection method
        phi = lambda alpha: f(x + alpha * direction)
        res = minimize_scalar(phi, bracket=(0,1), method='Brent')
        stepsize = res.x
        
        # Update x
        x = x + stepsize * direction
        x_hist.append(x)
        values.append(f(x))
        
    return x, values, x_hist

# Newton's Method (NR) with the corrected Hessian
def newtons_method(f, grad_f, hess_f, x0, tol=1e-6, max_iter=100):
    x = x0
    it = 0
    values = []
    x_hist = [x0]
    
    while np.linalg.norm(grad_f(x)) > tol and it < max_iter:
        it += 1
        direction = np.linalg.solve(hess_f(x), -grad_f(x))
        
        # Line search not necessary for quadratic functions when using Newton's method
        stepsize = .2
        
        # Update x
        x = x + stepsize * direction
        x_hist.append(x)
        values.append(f(x))
        
    return x, values, x_hist

# Initial guess
x0 = np.array([-2, 2, 2])

# Perform Steepest Descent with Bisection
sd_solution, sd_values, sd_history = steepest_descent_bisection(f, grad_f, x0)
# Perform Newton's Method with the corrected Hessian
nr_solution, nr_values, nr_history = newtons_method(f, grad_f, hess_f, x0)


# Plot cost vs iteration for both algorithms
plt.figure(figsize=(12, 6))
plt.plot(sd_values, label='Steepest Descent with Bisection')
plt.plot(nr_values, label="Newton's Method")
plt.xlabel('Iteration')
plt.ylabel('Cost')
plt.title('Cost vs Iteration')
plt.legend()
plt.grid(True)
plt.savefig('cost_vs_iteration.pdf')

# Function to plot cost contours and solution path
def plot_contours(f, x_history, title, filename):
    # Create a grid of values for x1 and x2 (ignoring x3 since it's always zero in this problem)
    x1 = np.linspace(-4, 4, 400)
    x2 = np.linspace(-4, 4, 400)
    X1, X2 = np.meshgrid(x1, x2)
    Z = np.zeros_like(X1)
    
    for i in range(X1.shape[0]):
        for j in range(X1.shape[1]):
            Z[i, j] = f([X1[i, j], X2[i, j], 0])
    
    # Plotting the contours
    plt.figure(figsize=(8, 6))
    cp = plt.contour(X1, X2, Z, 50, cmap='viridis')
    plt.colorbar(cp)
    x_hist = np.array(x_history)
    plt.plot(x_hist[:, 0], x_hist[:, 1], '-o', color='red', label='Path')
    plt.scatter(x_hist[-1, 0], x_hist[-1, 1], color='red', label='Solution')
    plt.title(title)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend()
    
    # Save the plot to a PDF
    pdf_pages = PdfPages(filename)
    pdf_pages.savefig()
    plt.close()
    pdf_pages.close()

# Plot the contours and paths for the corrected Newton's method
plot_contours(f, nr_history, "Corrected Newton's Method: Cost Contours and Solution Path", 'nr_contours.pdf')
# Plot the contours and paths for the Steepest Descent with Bisection
plot_contours(f, sd_history, "Steepest Descent with Bisection: Cost Contours and Solution Path", 'sd_contours.pdf')

# Print solutions to console
print("Steepest Descent with Bisection: ", sd_solution)
print("Newton's Method: ", nr_solution)
