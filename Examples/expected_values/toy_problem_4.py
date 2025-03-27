# import numpy as np

# # Number of Monte Carlo samples
# N = 10000000  

# # Step 1: Sample X from Uniform(0,1)
# X_samples = np.random.uniform(0, 1, N)

# # Step 2: Sample Y given X ~ N(X, 1)
# Y_samples = np.random.normal(loc=X_samples, scale=1, size=N)

# # Step 3: Compute E[Y] as the mean of Y_samples
# E_Y_MCS = np.mean(Y_samples)

# print(f"Estimated E[Y] using Monte Carlo Simulation: {E_Y_MCS:.9f}")

##############################################################################################

# import numpy as np
# from scipy.stats import beta, weibull_min

# # 1. Define distribution parameters
# alpha, beta_param = 2.0, 5.0    # Example Beta parameters
# k, scale = 1.5, 2.0             # Example Weibull parameters

# # 2. Draw samples
# N = 1000000  # Adjust as needed
# X = beta.rvs(alpha, beta_param, size=N)          # X ~ Beta(alpha, beta_param)
# Y = weibull_min.rvs(k, scale=scale, size=N)      # Y ~ Weibull(k) with scale=scale

# # 3. Define your function f(x, y)
# def f(x, y):
#     # Example "complex" function; replace with your actual function
#     return np.sin(x) * np.exp(-y) + x**2 + y**0.5

# # 4. Compute f on each sample pair
# f_vals = f(X, Y)  # vectorized if f uses NumPy

# # 5. Take the sample mean
# mc_estimate = np.mean(f_vals)

# print(f"Monte Carlo estimate for E[f(X,Y)] = {mc_estimate}")


# import numpy as np
# from scipy.integrate import simps
# from scipy.special import gamma, beta as beta_func

# #------------------------------------------------------------------------------
# # 1. Define distributions
# #    Beta(α, β) for x in (0,1)
# #    Weibull(k, λ) for y in (0,∞)  (we'll truncate at y_max)
# #------------------------------------------------------------------------------

# def beta_pdf(x, alpha, beta):
#     """
#     Beta PDF = x^(alpha-1)*(1-x)^(beta-1) / B(alpha,beta) for x in (0,1).
#     """
#     # We'll clip x to [0,1] to avoid numeric issues, but in a perfect scenario
#     # you only evaluate where 0<x<1 anyway.
#     x = np.clip(x, 0, 1)
#     B = beta_func(alpha, beta)  # Beta function
#     return x**(alpha-1) * (1 - x)**(beta-1) / B

# def weibull_pdf(y, k, lam):
#     """
#     Weibull PDF = (k/λ) * (y/λ)^(k-1) * exp(-(y/λ)^k), for y>0.
#     """
#     # Clip y to [0,∞) for safety
#     y = np.clip(y, 0, None)
#     return (k / lam) * (y / lam)**(k - 1) * np.exp(-(y / lam)**k)

# #------------------------------------------------------------------------------
# # 2. Define the function f(x, y)
# #    Replace with your actual "complex" function.
# #------------------------------------------------------------------------------
# def f(x, y):
#     return np.sin(x) * np.exp(-y) + x**2 + np.sqrt(y)

# #------------------------------------------------------------------------------
# # 3. Set the parameters and integration grids
# #------------------------------------------------------------------------------

# # Integration limits
# x_min, x_max = 0.0, 1.0
# y_min, y_max = 0.0, 15.0  # Truncate Weibull at 15 (adjust as needed)

# Nx = 201  # Number of grid points in x-direction
# Ny = 201  # Number of grid points in y-direction

# x_vals = np.linspace(x_min, x_max, Nx)
# y_vals = np.linspace(y_min, y_max, Ny)

# # Create a meshgrid so we can compute the 2D integrand easily
# X_mesh, Y_mesh = np.meshgrid(x_vals, y_vals, indexing='xy')

# #------------------------------------------------------------------------------
# # 4. Compute the integrand f(x,y)*g(x)*h(y) on a 2D grid
# #------------------------------------------------------------------------------
# gX = beta_pdf(X_mesh, alpha, beta_param)
# hY = weibull_pdf(Y_mesh, k, scale)

# integrand_2d = f(X_mesh, Y_mesh) * gX * hY

# #------------------------------------------------------------------------------
# # 5. Integrate in y for each x (Simpson's rule), then integrate that in x
# #------------------------------------------------------------------------------

# # 5a. Integrate over y, for each x. We get an array of shape (Nx,)
# partial_integrals = np.zeros(Nx)
# for i in range(Nx):
#     # integrand_2d[:, i] would be if we used indexing='ij', but we used 'xy',
#     # so the "rows" are y-values. 
#     # integrand_2d[i, :]  is the row at x_i if indexing='ij'. 
#     # Because we used indexing='xy', each row is a fixed x, so we want i-th row:
#     # Actually, let's check carefully:
#     #   X_mesh[i, j] = x_i
#     #   Y_mesh[i, j] = y_j   if indexing='ij'
#     # BUT we used 'xy', so
#     #   X_mesh[j, i] = x_i
#     #   Y_mesh[j, i] = y_j
#     # => the "j dimension" is y, the "i dimension" is x
#     # => integrand_2d[j, i] is the integrand at (x_i, y_j).
#     # So for a given x_i, we want integrand_2d[:, i].
#     y_slice = integrand_2d[:, i]
#     partial_integrals[i] = simps(y_slice, x=y_vals)

# # 5b. Now integrate partial_integrals over x
# expectation = simps(partial_integrals, x=x_vals)

# print(f"Simpson's rule 2D integral = {expectation:0.6f}")


# ########################################################################################

# import numpy as np
# from scipy.stats import beta, weibull_min

# # Distribution parameters
# alpha, beta_param = 2.0, 5.0
# k, scale = 1.5, 2.0

# # Probability function p(y) -> in [0,1]
# def p_func(y):
#     # Example: Sigmoid-ish shape
#     return 1.0 / (1.0 + np.exp(-(y - 3.0)))

# # Deterministic function h(x)
# def h_func(x):
#     return np.sin(x) + x**2

# # Constant c
# c = 10.0

# # 1. Sample X and Y
# N = 100000000
# X_samples = beta.rvs(alpha, beta_param, size=N)
# Y_samples = weibull_min.rvs(k, scale=scale, size=N)

# # 2. For each Y_i, compute p(y_i) and flip a Bernoulli
# p_vals = p_func(Y_samples)               # shape = (N,)
# bernoulli_draws = (np.random.rand(N) < p_vals).astype(int)

# # 3. Compute f_i
# h_vals = h_func(X_samples)
# f_vals = np.where(bernoulli_draws == 1, h_vals, c)

# # 4. Monte Carlo estimate
# mc_estimate = f_vals.mean()



# from scipy.integrate import simpson as simps
# from scipy.special import beta as beta_func
# from scipy.stats import weibull_min

# # ------------------------------------------------------------------------
# # 1) Define PDFs for Beta and Weibull
# # ------------------------------------------------------------------------
# def beta_pdf(x, alpha, beta):
#     """
#     Beta PDF on x in [0,1].
#     """
#     # Clip to avoid numerical issues (not strictly necessary if x is in [0,1])
#     x = np.clip(x, 0, 1)
#     B = beta_func(alpha, beta)  # Beta function B(α,β)
#     return x**(alpha-1) * (1 - x)**(beta-1) / B

# def weibull_pdf(y, k, lam):
#     """
#     Weibull PDF on y >= 0.
#     """
#     y = np.clip(y, 0, None)
#     return (k / lam) * (y / lam)**(k - 1) * np.exp(- (y / lam)**k)

# x_min, x_max = 0.0, 1.0
# y_min, y_max = 0.0, 50.0      # Truncate Weibull at y=15
# Nx = 4001
# Ny = 4001
# x_vals = np.linspace(x_min, x_max, Nx)
# y_vals = np.linspace(y_min, y_max, Ny)
# X_mesh, Y_mesh = np.meshgrid(x_vals, y_vals, indexing='xy')

# # Precompute PDFs
# gX = beta_pdf(X_mesh, alpha, beta_param)
# gY = weibull_pdf(Y_mesh, k, scale)
# f_mesh = p_func(Y_mesh) * h_func(X_mesh) + (1.0 - p_func(Y_mesh)) * c
# integrand_2d = f_mesh * gX * gY
# partial_integrals = np.zeros(Nx)
# for i in range(Nx):
#     partial_integrals[i] = simps(y=integrand_2d[:, i], x=y_vals)
# expectation = simps(y=partial_integrals, x=x_vals)


# print(f"Monte Carlo estimate = {mc_estimate:.6f}")
# print(f"Numerical integral via Simpson's rule = {expectation:.6f}")
# print(f"Diff = {mc_estimate - expectation:.6f}")

#############################################################################################

# import numpy as np
# from scipy.stats import beta, weibull_min
# import matplotlib.pyplot as plt

# # -----------------------------
# # 1) Define random distributions
# # -----------------------------
# alpha, beta_param = 2.0, 5.0      # for Beta distribution
# k_shape, lam_scale = 2.0, 1.0     # for Weibull(k=2, λ=1)

# # Beta PDF:  support x in [0,1]
# # Weibull PDF: support y in [0,∞]
# # For convenience, use scipy.stats objects:
# distX = beta(alpha, beta_param)
# distY = weibull_min(k_shape, scale=lam_scale)

# # -----------------------------
# # 2) Define the success function Z(y)
# #    (only depends on y)
# # -----------------------------
# def Z(y):
#     # Example: Z(y) = 1 - exp(-0.5*y)
#     # Feel free to pick any 0-1 valued function
#     return 1.0 - np.exp(-0.5*y)

# # -----------------------------
# # 3) Define E(x), C(x)
# #    (energy level if success or failure)
# # -----------------------------
# def E_func(x):
#     # Example: E(x) = 10*x - 3
#     return 10.0*x - 3.0

# def C_func(x):
#     # Example: always -2, ignoring x
#     return -2.0

# # -----------------------------
# # 4) Define V(s)
# #    (maps continuous energy -> discrete, with negative→0)
# # -----------------------------
# def V(s):
#     # Clip at 0 and then floor to integer
#     # e.g. V(s)=0 if s<0, otherwise floor(s)
#     if s < 0:
#         return 0
#     else:
#         return int(np.floor(s))

# # -----------------------------
# # 5) Numerical "Analytical" Approx: 2D integration
# #    (We do a grid approach for demonstration.)
# # -----------------------------
# # We'll truncate y on [0,5] as a rough upper bound
# # (Weibull with shape=2 can have support well beyond 5,
# #  but let's keep it short for example. Increase if needed.)
# x_vals = np.linspace(0, 1, 400)
# y_vals = np.linspace(0, 25, 400)
# dx = x_vals[1] - x_vals[0]
# dy = y_vals[1] - y_vals[0]

# # Precompute PDF values on grid
# fX = distX.pdf(x_vals)
# fY = distY.pdf(y_vals)

# expected_val_grid = 0.0

# for i, x in enumerate(x_vals):
#     valE = V(E_func(x))
#     valC = V(C_func(x))
#     for j, y in enumerate(y_vals):
#         p = Z(y)*valE + (1.0 - Z(y))*valC
#         expected_val_grid += p * fX[i] * fY[j] * dx * dy

# # This is our numerical approximation for the double integral:
# analytical_estimate = expected_val_grid

# # -----------------------------
# # 6) Monte Carlo Simulation
# # -----------------------------
# N = 10_000_00  # e.g. 1e6 draws
# X_samp = distX.rvs(size=N)
# Y_samp = distY.rvs(size=N)

# # For each (x,y), toss a Bernoulli with probability Z(y)
# # Then compute S' = E(x) or C(x), and apply V.
# u = np.random.rand(N)
# success_flags = (u < Z(Y_samp))
# Sprime = np.where(success_flags, E_func(X_samp), C_func(X_samp))

# # Vectorize V
# V_vectorized = np.vectorize(V)
# vals = V_vectorized(Sprime)

# mcs_estimate = np.mean(vals)

# # -----------------------------
# # 7) Print Results
# # -----------------------------
# print(f"Numerical-integration estimate of E[V] = {analytical_estimate:.8f}")
# print(f"Monte Carlo estimate of E[V]            = {mcs_estimate:.8f}")
# print(f"Difference                              = {abs(analytical_estimate - mcs_estimate):.8f}")

# # Optional: Show that the estimates are close if N is large enough

####################################################################################################################

import numpy as np
from scipy.stats import beta, weibull_min

# -------------------------------------------------
# 1) Define Distributions for X and Y
# -------------------------------------------------
#   X ~ Beta(alpha=2, beta=5), for example
#   Y ~ Weibull(k=2, lambda=1)
distX = beta(a=2.0, b=5.0)
distY = weibull_min(c=2.0, scale=5.0)

# -------------------------------------------------
# 2) Piecewise Z(y,m,a)
#    Each combo of m, a ∈ {0,1} can have a different formula.
#    Adapt to your real problem logic.
# -------------------------------------------------
def Z(y, m, a):
    """
    Return success probability for given (y, m, a).
    We'll define four toy cases for demonstration:
    (m=0,a=0): Z=0
    (m=0,a=1): Z = min(y/5, 1)
    (m=1,a=0): Z=0.5
    (m=1,a=1): Z=1 - exp(-y)
    """
    if m == 0 and a == 0:
        return 1.0
    elif m == 0 and a == 1:
        return np.minimum(2.0/(y+0.001), 1.0)
    elif m == 1 and a == 0:
        return np.exp(-.3*y)
    elif m == 1 and a == 1:
        return np.exp(-.25*y)
    # Fallback
    return 0.0

# -------------------------------------------------
# 3) Define parameters: s, e_r, C
#    Then E = s - e_r + X  (random if X is random)
# -------------------------------------------------
s = 10.0
e_r = 3.0
C = -2.0  # Failure leads to a constant negative state

def E_func(x):
    """Compute E = s - e_r + x."""
    return s - e_r + x*10.

# -------------------------------------------------
# 4) Value function V(s')
#    Negative -> 0, else floor
# -------------------------------------------------
def V(state):
    return 0 if state < 0 else int(np.floor(state))

# -------------------------------------------------
# 5) Numerical Integration
#    We'll do a 2D approach over X in [0,1], Y in [0,y_max].
#    Then we handle piecewise Z by plugging in m,a in each loop.
# -------------------------------------------------
num_points_x = 800
num_points_y = 800

# For X ~ Beta(2,5), domain is [0,1].
x_vals = np.linspace(0, 1, num_points_x)
dx = x_vals[1] - x_vals[0]
fX_vals = distX.pdf(x_vals)

# For Y ~ Weibull(2,1), domain is [0,∞), but we truncate:
y_max = 100.0
y_vals = np.linspace(0, y_max, num_points_y)
dy = y_vals[1] - y_vals[0]
fY_vals = distY.pdf(y_vals)

# -------------------------------------------------
# 6) Monte Carlo Setup
# -------------------------------------------------
N = 3000000  # number of samples

# We'll evaluate for m,a ∈ {0,1}×{0,1}.
m_values = [0, 1]
a_values = [0, 1]

print("m, a | Numerical_2D_Int  | MonteCarlo   |  Difference ")
print("-"*58)

for m_ in m_values:
    for a_ in a_values:
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # (A) Numerical 2D Integration
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        integral_sum = 0.0
        
        for i, x in enumerate(x_vals):
            e_x = E_func(x)    # s - e_r + x
            v_e = V(e_x)       # value for success
            for j, y in enumerate(y_vals):
                p_succ = Z(y, m_, a_)
                # integrand = [ p_succ*V(E) + (1-p_succ)*V(C ) ] * fX(x)*fY(y)
                integrand = p_succ * v_e + (1.0 - p_succ)*V(C)
                integral_sum += integrand * fX_vals[i] * fY_vals[j] * dx * dy
        
        numerical_est = integral_sum
        
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # (B) Monte Carlo
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # 1) Sample X, Y
        X_samp = distX.rvs(size=N)
        Y_samp = distY.rvs(size=N)
        
        # 2) Compute success probability for each (Y_i)
        p_succ_array = Z(Y_samp, m_, a_)
        
        # 3) Bernoulli draw
        u = np.random.rand(N)
        success_flags = (u < p_succ_array)
        
        # 4) E or C
        E_values = E_func(X_samp)    # array of s-e_r+X_i
        Sprime = np.where(success_flags, E_values, C)
        
        # 5) Apply V
        V_vectorized = np.vectorize(V)
        vals = V_vectorized(Sprime)
        
        mcs_est = np.mean(vals)
        
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Compare
        diff = abs(numerical_est - mcs_est)
        print(f"{m_}, {a_} |    {numerical_est:9.8f}  | {mcs_est:9.8f}  |  {diff:9.8f}")
