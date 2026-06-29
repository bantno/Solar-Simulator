import numpy as np
from scipy.integrate import quad, simpson
from scipy.stats import weibull_min


def compute_ps(alpha_0, alpha_1, lambda_param, k):
    """Compute P(S = 1) for a system where P(S = 1 | W = w) depends on a Weibull distribution of W.

    Args:
        alpha_0: Intercept for the sigmoid function.
        alpha_1: Coefficient for the sigmoid function (depends on W).
        lambda_param: Scale parameter of the Weibull distribution.
        k: Shape parameter of the Weibull distribution.

    Returns:
        P(S = 1): The marginal probability.
    """

    # Define the conditional probability P(S = 1 | W = w)
    def conditional_ps(w):
        return 1 - 1 / (1 + np.exp(-(alpha_0 + alpha_1 * w)))

    # Define the Weibull PDF for W
    def weibull_pdf(w):
        return weibull_min.pdf(w, k, scale=lambda_param)

    # Define the integrand: P(S = 1 | W = w) * f_W(w)
    def integrand(w):
        return conditional_ps(w) * weibull_pdf(w)

    # Perform numerical integration from 0 to infinity
    ps, _ = quad(integrand, 0, np.inf)  # _ gives the error estimate, which we ignore
    return ps


def compute_ps_simpsons(alpha_0, alpha_1, lambda_param, k, w_max=10, n_points=1000):
    """Compute P(S = 1) using Simpson's rule for a system where P(S = 1 | W = w)
    depends on a Weibull distribution of W.

    Args:
        alpha_0: Intercept for the sigmoid function.
        alpha_1: Coefficient for the sigmoid function (depends on W).
        lambda_param: Scale parameter of the Weibull distribution.
        k: Shape parameter of the Weibull distribution.
        w_max: Upper limit for approximating the range of W.
        n_points: Number of points to sample in the range [0, w_max].

    Returns:
        P(S = 1): The marginal probability.
    """
    # Create a range of W values from 0 to w_max
    w_values = np.linspace(0.001, w_max, n_points)

    # Define the conditional probability P(S = 1 | W = w)
    def conditional_ps(w):
        return 1 - 1 / (1 + np.exp(-(alpha_0 + alpha_1 * w)))

    # Define the Weibull PDF for W
    def weibull_pdf(w):
        # return weibull_min.pdf(w, k, scale=lambda_param)
        return (
            (k / lambda_param) * (w / lambda_param) ** (k - 1) * np.exp(-((w / lambda_param) ** k))
        )

    # Compute the integrand for each value of W
    integrand_values = conditional_ps(w_values) * weibull_pdf(w_values)

    # Perform numerical integration using Simpson's rule
    ps = simpson(y=integrand_values, x=w_values)
    return ps


# Example usage
alpha_0 = -10.0  # Intercept for sigmoid
alpha_1 = 0.25  # Coefficient for sigmoid
lambda_param = 20.0  # Scale parameter for Weibull
k = 10.0  # Shape parameter for Weibull

ps = compute_ps(alpha_0, alpha_1, lambda_param, k)
print(f"quad   : P(S = 1): {ps}")

w_max = 100  # Approximate upper limit for W
n_points = 501  # Number of sample points

ps = compute_ps_simpsons(alpha_0, alpha_1, lambda_param, k, w_max, n_points)
print(f"simspon: P(S = 1): {ps}")
