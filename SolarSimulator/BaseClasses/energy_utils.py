# energy_utils.py

def soc_to_joules(soc, battery_capacity_wh):
    """
    Convert a state-of-charge percentage to energy in Joules.
    
    Parameters:
        soc: State of charge (percentage).
        battery_capacity_wh: Battery capacity in watt-hours.
        
    Returns:
        Energy in Joules (integer).
    """
    joules = soc / 100 * battery_capacity_wh * 3600
    return round(joules)

def joules_to_soc(joules, battery_capacity_wh):
    """
    Convert energy in Joules to a state-of-charge percentage.
    
    Parameters:
        joules: Energy in Joules.
        battery_capacity_wh: Battery capacity in watt-hours.
        
    Returns:
        SOC as a percentage.
    """
    soc = (joules / (battery_capacity_wh * 3600)) * 100
    return soc

def round_to_precision(value, precision):
    """
    Round a value to the nearest multiple of the given precision.
    
    Parameters:
        value: The number to round.
        precision: The precision step.
        
    Returns:
        Rounded value.
    """
    return round(value / precision) * precision
