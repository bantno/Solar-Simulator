import pandas as pd
from tqdm import tqdm

# data = pd.read_pickle(rf"Data\EXPECTED_DATA\data_expected_lat0_lon-90_15min.pkl")
# print(data)

import numpy as np
from scipy.stats import beta
import scipy.stats as stats

data = pd.read_pickle(rf"Data\HISTORICAL_DATA\data_0_-90")

# Your original data: ensure they lie in [0, 1] for a beta distribution.
df = data
shape_rses = []
scale_rses = []
shape_means = []
scale_means = []
scale_ses = []
shape_ses = []


for month in tqdm([1,3,6,9]):
    for day in [1]:
        for hour in [6]:
            target_month = month
            target_day = day
            target_hour = hour
            filtered_df = df[(df.index.month == target_month) & 
                            (df.index.day == target_day) & 
                            (df.index.hour == target_hour)]


            n_bootstrap = 1000
            bootstrap_alphas = []
            bootstrap_betas = []
            data = np.array(filtered_df['wind_speed_10m'])



            # Assume `data` is your observed dataset
            n_bootstrap = 5000  # Number of bootstrap samples

            bootstrap_shapes = []
            bootstrap_scales = []

            for _ in range(n_bootstrap):
                # Create a bootstrap sample (sample with replacement)
                sample = np.random.choice(data, size=len(data), replace=True)
                
                # Fit Weibull distribution to the sample; scipy.stats uses a shape-scale parameterization
                shape, loc, scale = stats.weibull_min.fit(sample, floc=0)  # Fix loc=0 if data is strictly positive

                bootstrap_shapes.append(shape)
                bootstrap_scales.append(scale)

            # Convert to numpy arrays
            bootstrap_shapes = np.array(bootstrap_shapes)
            bootstrap_scales = np.array(bootstrap_scales)

            # Calculate summary statistics (mean and standard error)
            shape_mean = np.mean(bootstrap_shapes)
            scale_mean = np.mean(bootstrap_scales)
            shape_se = np.std(bootstrap_shapes)
            scale_se = np.std(bootstrap_scales)

            shape_means.append(shape_mean)
            scale_means.append(scale_mean)
            shape_ses.append(shape_se)
            scale_ses.append(scale_se)
            shape_rses.append(shape_se/shape_mean)
            scale_rses.append(scale_se/scale_mean)

# Print results
print("Weibull shape (k) mean estimate:", np.mean(shape_means))
print("Weibull scale (λ) mean estimate:", np.mean(scale_means))
print("Weibull shape standard error:", np.mean(shape_ses))
print("Weibull scale standard error:", np.mean(scale_se))
print("Weibull shape relative standard error:", np.mean(shape_rses))
print("Weibull scale relative standard error:", np.mean(scale_rses))

print("Weibull shape (k) mean estimate:", (shape_means))
print("Weibull scale (λ) mean estimate:", (scale_means))
print("Weibull shape standard error:", (shape_ses))
print("Weibull scale standard error:", (scale_se))
print("Weibull shape relative standard error:", (shape_rses))
print("Weibull scale relative standard error:", (scale_rses))