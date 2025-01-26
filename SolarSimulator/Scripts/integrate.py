import numpy as np
from scipy.integrate import quad

# Define the PDF f_k(x) as a function
def f_k(x):
    # Example: A standard normal distribution as a placeholder
    return (1 / np.sqrt(2 * np.pi)) * np.exp(-x**2 / 2)

# Define the limits of the integral
delta_alpha = 1  # Replace with the actual value of δα
upper_limit = np.inf

# Define the integrand x * f_k(x)
def integrand(x):
    return x * f_k(x)

# Compute the integral
result, error = quad(integrand, delta_alpha, upper_limit)

# Print the result
print(f"Integral value: {result}")
print(f"Estimated error: {error}")