# result_formatter.py

def format_simulation_result(result, algo, expected_data, save_history=False):
    """
    Format simulation result into a structured dictionary.
    
    Parameters:
        result: Tuple of simulation outputs.
        algo: Algorithm name used in the simulation.
        expected_data: DataFrame with expected weather data.
        save_history: Boolean indicating whether full history should be included.
        
    Returns:
        Dictionary containing formatted simulation results.
    """
    if save_history:
        (reward, last_step, state_history, action_history, failure_prob_history,
         solar_history, wind_history, whale_history, flight_minutes, is_failure, failure_type) = result
        return {
            "Reward": reward,
            "LastStep": last_step,
            "Algorithm": algo,
            "ActionHistory": action_history.tolist() if hasattr(action_history, "tolist") else action_history,
            "FailureProbHistory": failure_prob_history.tolist() if hasattr(failure_prob_history, "tolist") else failure_prob_history,
            "StateHistory": state_history.tolist() if hasattr(state_history, "tolist") else state_history,
            "SolarHistory": solar_history,
            "ExpectedSolarHistory": expected_data["expected_solar_rad"].values.tolist(),
            "WindHistory": wind_history,
            "ExpectedWindHistory": expected_data["expected_wind_speed"].values.tolist(),
            "WhaleHistory": whale_history,
            "FlightHours": flight_minutes / 60,
            "Failure": is_failure,
            "FailureType": failure_type,
        }
    else:
        reward, last_step, flight_minutes, is_failure, failure_type = result
        return {
            "Reward": reward,
            "Algorithm": algo,
            "LastStep": last_step,
            "FlightHours": flight_minutes / 60,
            "Failure": is_failure,
            "FailureType": failure_type,
        }
