import numpy as np
from scipy.stats import beta as beta_dist
from scipy.stats import weibull_min
from scipy.integrate import dblquad

# ---------------------------------------------------------------------
# 1. USER-DEFINED PARAMETERS & FUNCTIONS
# ---------------------------------------------------------------------

# Distributions
alpha, beta_ = 2.0, 5.0   # Beta distribution parameters
k, lam = 2.0, 1.0         # Weibull distribution parameters (shape=k, scale=lam)

# The "probabilistic" definition of Z:
#   Z = f(x,y)   w.p. p(x,y)
#   Z = c        w.p. 1 - p(x,y)

def p_xy(x, y):
    """Probability that Z = f(x,y). Must be in [0,1]."""
    return np.clip(0.25*x+.4*y,0.,1.)  # Example: use x as the probability

def f_xy(x, y):
    """The function f(x,y)."""
    return x + y  # Example choice

c = 1.0  # Constant fallback value for Z

# ---------------------------------------------------------------------
# 2. ANALYTICAL SOLUTION VIA NUMERICAL INTEGRATION
# ---------------------------------------------------------------------
# E[Z] = ∫∫ [ p(x,y)*f(x,y) + (1 - p(x,y))*c ] * g(x)*h(y) dx dy

# Beta PDF on [0, 1]
def g_x(x):
    return beta_dist.pdf(x, alpha, beta_)

# Weibull PDF on [0, ∞)
def h_y(y):
    return weibull_min.pdf(y, k, scale=lam)

def integrand(y, x):
    """Return [p(x,y)*f(x,y) + (1 - p(x,y))*c] * g(x)*h(y).
    Note dblquad integrates in the order: integrand(y, x).
    """
    return (p_xy(x, y)*f_xy(x, y) + (1 - p_xy(x, y))*c) * g_x(x) * h_y(y)

def compute_EZ_analytical():
    """Numerically integrates the above expression:
    x in [0, 1],  y in [0, ∞).
    """
    result, abserr = dblquad(
        integrand,
        0,    # x lower limit
        1,    # x upper limit
        lambda x: 0,       # y lower limit
        lambda x: np.inf   # y upper limit
    )
    return result

# ---------------------------------------------------------------------
# 3. MONTE CARLO SIMULATION (MCS)
# ---------------------------------------------------------------------

def compute_EZ_mcs(N=10_000_00):
    """Estimates E[Z] by:

    1) Drawing N samples of (X, Y).
    2) For each (x_i, y_i), draw Bernoulli(p(x_i,y_i)).
    3) If Bernoulli=1, Z_i = f(x_i,y_i), else Z_i = c.
    4) Average all Z_i.
    """
    # 1) Sample X ~ Beta(alpha, beta_)
    X_samples = beta_dist.rvs(alpha, beta_, size=N)
    
    # 2) Sample Y ~ Weibull(k, scale=lam)
    Y_samples = weibull_min.rvs(k, scale=lam, size=N)
    
    # 3) For each pair (x_i, y_i), compute Z_i
    #    We flip a Bernoulli with parameter p_xy(x_i, y_i).
    U = np.random.rand(N)  # uniform(0,1) for Bernoulli
    p_vals = p_xy(X_samples, Y_samples)  # vectorized if p_xy is vector-friendly
    # If p_vals is not vectorized, you can do list comprehension
    # p_vals = np.array([p_xy(x_i, y_i) for x_i, y_i in zip(X_samples, Y_samples)])
    
    Z_samples = np.where(U < p_vals, f_xy(X_samples, Y_samples), c)
    
    # 4) Return average
    return Z_samples.mean()

# ---------------------------------------------------------------------
# 4. MAIN: COMPARE ANALYTICAL VS. MCS
# ---------------------------------------------------------------------
if __name__ == "__main__":
    # (A) Analytical (via numerical integration)
    EZ_analytical = compute_EZ_analytical()
    
    # (B) Monte Carlo
    N_samples = 1_000_000
    EZ_mcs = compute_EZ_mcs(N_samples)
    
    # Print results
    print(f"Analytical E[Z] (via dblquad)      = {EZ_analytical:.6f}")
    print(f"Monte Carlo E[Z] (N={N_samples})    = {EZ_mcs:.6f}")
    print(f"Absolute Error: {np.abs(EZ_analytical - EZ_mcs):.6f}")
