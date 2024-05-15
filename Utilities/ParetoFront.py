import numpy as np
import matplotlib.pyplot as plt

class ParetoFront:
    def __init__(self, objective_functions, bounds):
        self.objective_functions = objective_functions
        self.bounds = bounds

    def is_dominated(self, point, population):
        """
        Check if a point is dominated by any point in the population.
        """
        for p in population:
            if all(p[i] <= point[i] for i in range(len(point))) and any(p[j] < point[j] for j in range(len(point))):
                return True
        return False

    def lhs_sampling(self, n_samples):
        """
        Latin Hypercube Sampling for Pareto Front.
        """
        samples = np.zeros((n_samples, len(self.bounds)))
        for i in range(len(self.bounds)):
            samples[:, i] = np.random.uniform(self.bounds[i][0], self.bounds[i][1], n_samples)
        for i in range(len(self.bounds)):
            np.random.shuffle(samples[:, i])
        return samples.tolist()

    def generate_pareto_front(self, n_samples):
        """
        Construct Pareto front using LHS.
        """
        pareto_front = []
        samples = self.lhs_sampling(n_samples)
        for point in samples:
            if not self.is_dominated(self.objective_functions(point), pareto_front):
                pareto_front.append(self.objective_functions(point))
        return pareto_front

# Example usage:

# Define the bounds for each decision variable
bounds = [(-5, 5), (-5, 5)]  # Example bounds for a 2D problem

# Define the objective functions
def objective_functions(x):
    """
    Define your objective functions here.
    For example, for a two-objective problem:

    """
    f1 = x[0]**2
    f2 = (x[0]-2)**2
    return [f1, f2]

# Create an instance of ParetoFront
pareto_front = ParetoFront(objective_functions, bounds)

# Number of samples for Latin Hypercube Sampling
n_samples = 100

# Calculate the Pareto front using LHS
pf = pareto_front.generate_pareto_front(n_samples)

# Extract the objective values for plotting
pf = np.array(pf)

# Plot the Pareto front
plt.scatter(pf[:, 0], pf[:, 1], marker='o', color='b', label='Pareto Front (LHS)')
plt.xlim(-1,5)
plt.ylim(-1,5)
plt.xlabel('Objective 1')
plt.ylabel('Objective 2')
plt.title('Pareto Front using Latin Hypercube Sampling')
plt.legend()
plt.grid(True)
plt.show()
