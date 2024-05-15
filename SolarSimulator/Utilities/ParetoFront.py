import numpy as np
import matplotlib.pyplot as plt

class Pareto:
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

    def generate_pareto_front(self, n_samples,plane):
        """
        Construct Pareto front and non-dominated points using LHS.
        """
        pareto_front = []
        non_dominated_points = []
        samples = self.lhs_sampling(n_samples)
        for point in samples:
            obj_point = self.objective_functions(point,plane)
            if not self.is_dominated(obj_point, pareto_front):
                pareto_front.append(obj_point)
            else: 
                non_dominated_points.append(obj_point)
        return pareto_front, non_dominated_points



