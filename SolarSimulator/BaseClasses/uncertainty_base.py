import numpy as np
from scipy.stats import norm, multivariate_normal
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

class Uncertainty:
    def __init__(self, tmy_data, uncertainties, start_date, time_step, latitude):
        """
        Initialize the Uncertainty class with TMY data, uncertainties, and time information.
        
        :param tmy_data: dict with keys 'ghi', 'dhi', 'dni' containing data for a year at the simulation time step
        :param uncertainties: dict with keys 'ghi', 'dhi', 'dni' containing uncertainty values (as percentages)
        :param start_date: datetime object representing the start of the simulation
        :param time_step: timedelta object representing the time step of the simulation
        :param latitude: float, latitude of the location in degrees
        """
        self.tmy_data = tmy_data
        self.uncertainties = uncertainties
        self.start_date = start_date
        self.time_step = time_step
        self.latitude = np.radians(latitude)
        self.correlation_matrix = np.array([
            [1.0, 0.7, 0.3],
            [0.7, 1.0, -0.3],
            [0.3, -0.3, 1.0]
        ])

    def generate_time_series(self, duration):
        """
        Generate time series data for ghi, dhi, and dni with daily variations.
        
        :param duration: timedelta object representing the duration of the simulation
        :return: dict with keys 'timestamp', 'ghi', 'dhi', 'dni', 'ghi_true', 'dhi_true', 'dni_true'
        """
        num_steps = int(duration / self.time_step)
        timestamps = [self.start_date + i * self.time_step for i in range(num_steps)]
        
        ghi_series, dhi_series, dni_series = [], [], []
        ghi_true, dhi_true, dni_true = [], [], []
        
        # Generate daily variation factors
        num_days = (duration.days + 1)  # +1 to ensure we cover partial days
        daily_factors = self.generate_daily_factors(num_days)
        
        for timestamp in timestamps:
            index = int((timestamp - self.start_date) / self.time_step) % len(self.tmy_data['ghi'])
            day_index = (timestamp - self.start_date).days
            
            ghi_base = self.tmy_data['ghi'][index]
            dhi_base = self.tmy_data['dhi'][index]
            dni_base = self.tmy_data['dni'][index]
            
            if ghi_base == 0:
                ghi, dhi, dni = 0, 0, 0
            else:
                solar_zenith = self.calculate_solar_zenith(timestamp)
                ghi, dhi, dni = self.generate_consistent_sample(
                    ghi_base, dhi_base, dni_base,
                    daily_factors['ghi'][day_index],
                    daily_factors['dhi'][day_index],
                    daily_factors['dni'][day_index],
                    solar_zenith
                )
            
            ghi_series.append(ghi)
            dhi_series.append(dhi)
            dni_series.append(dni)
            ghi_true.append(ghi_base)
            dhi_true.append(dhi_base)
            dni_true.append(dni_base)
        
        return {
            'timestamp': timestamps,
            'ghi': ghi_series,
            'dhi': dhi_series,
            'dni': dni_series,
            'ghi_true': ghi_true,
            'dhi_true': dhi_true,
            'dni_true': dni_true
        }

    def generate_daily_factors(self, num_days):
        """
        Generate daily variation factors for ghi, dhi, and dni.
        
        :param num_days: int, number of days to generate factors for
        :return: dict with keys 'ghi', 'dhi', 'dni' containing daily factors
        """
        mean = [1, 1, 1]
        cov = np.diag([self.uncertainties[k]/100 for k in ['ghi', 'dhi', 'dni']])
        cov = np.dot(np.dot(cov, self.correlation_matrix), cov)
        
        daily_factors = multivariate_normal.rvs(mean=mean, cov=cov, size=num_days)
        return {
            'ghi': np.maximum(daily_factors[:, 0], 0),
            'dhi': np.maximum(daily_factors[:, 1], 0),
            'dni': np.maximum(daily_factors[:, 2], 0)
        }

    def generate_consistent_sample(self, ghi_base, dhi_base, dni_base, ghi_factor, dhi_factor, dni_factor, solar_zenith):
        """
        Generate a physically consistent sample of ghi, dhi, and dni.
        
        :param ghi_base, dhi_base, dni_base: float, base values from TMY data
        :param ghi_factor, dhi_factor, dni_factor: float, daily variation factors
        :param solar_zenith: float, solar zenith angle in radians
        :return: tuple of (ghi, dhi, dni) values
        """
        cos_zenith = np.cos(solar_zenith)
        
        # Apply factors to base values
        ghi = ghi_base * ghi_factor
        dhi = dhi_base * dhi_factor
        dni = dni_base * dni_factor
        
        # Ensure physical consistency
        ghi = max(dhi, ghi)  # GHI should be at least as large as DHI
        dni = max(0, (ghi - dhi) / cos_zenith)  # Recalculate DNI to ensure consistency
        
        return ghi, dhi, dni

    def calculate_solar_zenith(self, timestamp):
        """
        Calculate the solar zenith angle for a given timestamp and latitude.
        This is a simplified calculation and doesn't account for longitude or time zones.
        
        :param timestamp: datetime object
        :return: float, solar zenith angle in radians
        """
        day_of_year = timestamp.timetuple().tm_yday
        declination = 23.45 * np.sin(np.radians(360/365 * (day_of_year - 81)))
        declination = np.radians(declination)
        
        hour_angle = (timestamp.hour + timestamp.minute / 60) * 15 - 180
        hour_angle = np.radians(hour_angle)
        
        cos_zenith = (np.sin(self.latitude) * np.sin(declination) +
                      np.cos(self.latitude) * np.cos(declination) * np.cos(hour_angle))
        return np.arccos(cos_zenith)

    def plot_time_series(self, data, days=7):
        """
        Plot time series data for ghi, dhi, and dni, including true TMY data.
        
        :param data: dict with keys 'timestamp', 'ghi', 'dhi', 'dni', 'ghi_true', 'dhi_true', 'dni_true'
        :param days: int, number of days to plot
        """
        end_index = int(days * 24 * 3600 / self.time_step.total_seconds())
        
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 15))
        
        # ghi plot
        ax1.plot(data['timestamp'][:end_index], data['ghi'][:end_index], label='GHI (Simulated)', alpha=0.7)
        ax1.plot(data['timestamp'][:end_index], data['ghi_true'][:end_index], label='GHI (True)', linestyle='--')
        ax1.set_ylabel('GHI (W/m²)')
        ax1.legend()
        ax1.grid(True)
        
        # dhi plot
        ax2.plot(data['timestamp'][:end_index], data['dhi'][:end_index], label='DHI (Simulated)', alpha=0.7)
        ax2.plot(data['timestamp'][:end_index], data['dhi_true'][:end_index], label='DHI (True)', linestyle='--')
        ax2.set_ylabel('DHI (W/m²)')
        ax2.legend()
        ax2.grid(True)
        
        # dni plot
        ax3.plot(data['timestamp'][:end_index], data['dni'][:end_index], label='DNI (Simulated)', alpha=0.7)
        ax3.plot(data['timestamp'][:end_index], data['dni_true'][:end_index], label='DNI (True)', linestyle='--')
        ax3.set_ylabel('DNI (W/m²)')
        ax3.set_xlabel('Time')
        ax3.legend()
        ax3.grid(True)
        
        plt.suptitle(f'Simulated vs True Irradiance Data for {days} Days')
        plt.tight_layout()
        plt.show()

# Usage example
if __name__ == "__main__":
    # Example TMY data (you would typically load this from a file)
    # This example assumes 15-minute data for a full year (35040 data points)
    num_data_points = 35040
    tmy_data = {
        'ghi': [0] * num_data_points,
        'dhi': [0] * num_data_points,
        'dni': [0] * num_data_points
    }
    # Fill with some example data
    for i in range(num_data_points):
        hour = (i * 15 / 60) % 24
        day_of_year = i // (24 * 4)
        seasonal_factor = 1 + 0.2 * np.sin(2 * np.pi * day_of_year / 365)
        if 6 <= hour < 18:  # Daylight hours
            tmy_data['ghi'][i] = max(0, 800 * seasonal_factor * np.sin(np.pi * (hour - 6) / 12))
            tmy_data['dhi'][i] = max(0, 200 * seasonal_factor * np.sin(np.pi * (hour - 6) / 12))
            tmy_data['dni'][i] = max(0, 600 * seasonal_factor * np.sin(np.pi * (hour - 6) / 12))
        else:  # Night time
            tmy_data['ghi'][i] = 0
            tmy_data['dhi'][i] = 0
            tmy_data['dni'][i] = 0

    uncertainties = {
        'ghi': 5,  # 5% uncertainty
        'dhi': 7,  # 7% uncertainty
        'dni': 10  # 10% uncertainty
    }

    start_date = datetime(2024, 1, 1)
    time_step = timedelta(minutes=15)
    latitude = 40.0  # Example latitude (40 degrees North)

    uncertainty_model = Uncertainty(tmy_data, uncertainties, start_date, time_step, latitude)
    
    # Generate time series data for 30 days
    simulation_data = uncertainty_model.generate_time_series(timedelta(days=30))
    
    # Plot the first 7 days of data
    uncertainty_model.plot_time_series(simulation_data, days=7)
