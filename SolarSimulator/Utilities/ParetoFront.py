import numpy as np
import pandas as pd
from tqdm import tqdm
from paretoset import paretoset

class Pareto:
    def __init__(self, objective_functions, bounds):
        self.objective_functions = objective_functions
        self.bounds = bounds

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
        values = []
        samples = self.lhs_sampling(n_samples)

        for point in tqdm(samples):
            obj_point = self.objective_functions(point,plane)
            values.append(obj_point)
        samples = pd.DataFrame(samples)
        values = pd.DataFrame(values)

        mask = paretoset(values, sense=["max", "min"])
        return samples, values[mask], values



