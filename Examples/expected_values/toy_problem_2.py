import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta, weibull_min
from scipy.integrate import dblquad

# ------------------------------------------------------------
# 1. USER-DEFINED SETTINGS
# ------------------------------------------------------------

# Beta distribution parameters
alpha = 2.0
beta_ = 5.0  # renamed to beta_ to avoid conflict with 'beta' from scipy.stats

# Weibull distribution parameters
k = 2.0       # shape
lambda_ = 1.0 # scale

# Function f(x, y)
def f(x, y):
    return x**2 + y**3

# ------------------------------------------------------------
# 2. MONTE CARLO SIMULATION
# ------------------------------------------------------------

def estimate_EZ_monte_carlo(N=10_000):
    """
    Estimates E[f(X,Y)] via Monte Carlo simulation,
    given X ~ Beta(alpha, beta) and Y ~ Weibull(k, lambda).
    """
    # Sample from Beta(alpha, beta_)
    X_samples = beta.rvs(alpha, beta_, size=N)
    # Sample from Weibull(k, scale=lambda_)
    Y_samples = weibull_min.rvs(k, scale=lambda_, size=N)
    
    # Compute f(X_i, Y_i) for each sample
    Z_samples = f(X_samples, Y_samples)
    
    # Monte Carlo estimate is the sample average
    return np.mean(Z_samples)

# ------------------------------------------------------------
# 3. NUMERICAL INTEGRATION
# ------------------------------------------------------------
#
# E[Z] = ∫∫ f(x, y) g(x) h(y) dx dy
#
# where g(x) is the Beta PDF (defined on [0,1]) 
# and h(y) is the Weibull PDF (defined on [0,∞)).

from scipy.stats import beta as beta_dist, weibull_min as weibull_dist

def beta_pdf(x, alpha, beta_):
    return beta_dist.pdf(x, alpha, beta_)

def weibull_pdf(y, k, lambda_):
    return weibull_dist.pdf(y, k, scale=lambda_)

def integrand(y, x, alpha, beta_, k, lambda_):
    """
    integrand = f(x, y) * g(x) * h(y)
    The dblquad function signature for integrand is integrand(y, x).
    """
    return f(x, y) * beta_pdf(x, alpha, beta_) * weibull_pdf(y, k, lambda_)

def estimate_EZ_numerical_integration():
    """
    Numerically evaluates the double integral of f(x,y) * g(x) * h(y) dx dy.
    Uses scipy.integrate.dblquad over x in [0,1] and y in [0,∞).
    """
    # Inner integral: integrate over y from 0 to ∞
    # Outer integral: integrate over x from 0 to 1
    # Note that dblquad expects integrand(y, x).
    result, abserr = dblquad(
        integrand,
        0,   # x lower limit
        1,   # x upper limit
        lambda x: 0,        # y lower limit
        lambda x: np.inf,   # y upper limit
        args=(alpha, beta_, k, lambda_)
    )
    return result

# ------------------------------------------------------------
# 4. MAIN EXECUTION / EXAMPLE USAGE
# ------------------------------------------------------------
if __name__ == "__main__":
    # 4.1 Monte Carlo estimate
    N_samples = 100_000_000
    mc_estimate = estimate_EZ_monte_carlo(N_samples)
    
    # 4.2 Numerical integration
    numerical_estimate = estimate_EZ_numerical_integration()
    
    # Print results
    print(f"Monte Carlo estimate of E[Z] (N={N_samples}): {mc_estimate:.6f}")
    print(f"Numerical integration estimate of E[Z]:       {numerical_estimate:.6f}")
    print(f"Difference: {np.abs(mc_estimate - numerical_estimate):.6f}")
