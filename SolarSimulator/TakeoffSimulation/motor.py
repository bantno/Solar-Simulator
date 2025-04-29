import numpy as np

class Motor:
    """
    Motor model for HQ8040-3 based on static test data.
    
    Attributes:
        throttle_arr (np.ndarray): Throttle settings (%).
        thrust_g_arr  (np.ndarray): Thrust (g) at each throttle.
        power_W_arr   (np.ndarray): Power (W) at each throttle.
    """
    def __init__(self):
        # Throttle settings (%) 
        self.throttle_arr = np.array(
            [30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 
             80, 85, 90, 95, 100], dtype=float)
        # Corresponding static thrust (g)
        self.thrust_g_arr = np.array(
            [417.23, 511.31, 603.20, 695.81, 790.22,
             886.06, 998.05, 1149.32, 1325.18, 1508.82,
             1680.88, 1882.06, 2069.79, 2257.18, 2450.57],
            dtype=float)
        # Corresponding input power (W)
        self.power_W_arr = np.array(
            [78.22, 100.84, 124.50, 148.62, 172.83,
             197.01, 232.85, 270.95, 328.31, 394.95,
             461.31, 541.86, 624.86, 707.88, 801.00],
            dtype=float)

    def thrust(self, throttle: float, units: str = 'g') -> float:
        """
        Interpolate static thrust at a given throttle.

        Args:
            throttle (float): Throttle setting %.
            units (str): 'g' or 'N'.

        Returns:
            float: Thrust in desired units.
        """
        thr = np.clip(throttle, self.throttle_arr.min(), self.throttle_arr.max())
        thrust_g = np.interp(thr, self.throttle_arr, self.thrust_g_arr)
        if units == 'g':
            return thrust_g
        elif units == 'N':
            return thrust_g * 0.00980665
        else:
            raise ValueError("Units must be 'g' or 'N'")

    def power(self, throttle: float, units: str = 'W') -> float:
        """
        Interpolate input power at a given throttle.

        Args:
            throttle (float): Throttle setting %.
            units (str): 'W' or 'kW'.

        Returns:
            float: Power in desired units.
        """
        thr = np.clip(throttle, self.throttle_arr.min(), self.throttle_arr.max())
        p_W = np.interp(thr, self.throttle_arr, self.power_W_arr)
        if units == 'W':
            return p_W
        elif units == 'kW':
            return p_W / 1000.0
        else:
            raise ValueError("Units must be 'W' or 'kW'")

    def efficiency(self, throttle: float) -> float:
        """
        Compute efficiency (thrust per watt) at a given throttle.

        Args:
            throttle (float): Throttle setting %.

        Returns:
            float: Efficiency in g/W.
        """
        thrust_g = self.thrust(throttle, units='g')
        p_W = self.power(throttle, units='W')
        return thrust_g / p_W
