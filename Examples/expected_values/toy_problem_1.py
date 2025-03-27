import numpy as np
from scipy.stats import beta
from scipy.integrate import quad

# Define parameters for the Beta distribution: X ~ Beta(2, 5)
a, b = 20, 15

# Define the function h(x)
def h(x):
    return np.sqrt(x)

# Define the PDF of X using the Beta distribution
def g(x):
    return beta.pdf(x, a, b)

# Compute the theoretical expectation using numerical integration
numerical_expectation, integration_error = quad(lambda x: h(x) * g(x), 0, 1)

# Monte Carlo simulation
N = 1_000_000  # number of samples

# Sample X from the Beta distribution
X_samples = beta.rvs(a, b, size=N)

# For each sample, compute the probability p = h(X)
p_values = h(X_samples)

# Generate Y ~ Bernoulli(p) for each X sample
Y_samples = np.random.binomial(1, p_values)

# Calculate the empirical mean of Y
empirical_mean = np.mean(Y_samples)
print(f"Numerical integration E[Y]: {numerical_expectation:9.8f}")
print(f"Empirical simulation E[Y]: {empirical_mean:9.8f}")
print(f"Difference: {np.abs(numerical_expectation - empirical_mean):9.8f}")
