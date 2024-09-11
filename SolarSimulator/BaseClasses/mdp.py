import pandas as pd
import numpy as np
from tqdm import tqdm


class mdp:
    """
    Class representing a markov decision process problem
    """

    def __init__(self, plane, soc_increment, vehicle_states, max_stages, actions, stm):
        self.vehicle_states = vehicle_states
        self.plane = plane
        self.battery_capacity = self.plane.voltage*self.plane.capacity
        self.stm = stm
        self.states = self.create_states(soc_increment, vehicle_states)
        self.actions = actions
        self.prob=1

        self.create_ev_table(max_stages)
        

    @staticmethod
    def create_states(soc_increment: int, vehicle_states: list) -> list:
        """
        Generate a list of states based on state of charge (SoC) increments and specified vehicle states.

        This static method creates a list of states where each state is represented by a tuple
        containing the state of charge (SoC) and the state type. The SoC values range from 0 to 100
        in increments specified by the `soc_increment` parameter. The state types are provided
        by the `vehicle_states` parameter, allowing for flexibility in defining different vehicle states.

        Parameters:
            soc_increment (int): The increment value for state of charge. It must be a positive integer
                                that determines the step size between SoC values.
            vehicle_states (list): A list of state types (strings) to be used in the tuples. Each state
                                type will be combined with the SoC values to create the list of states.

        Returns:
            list: A list of tuples where each tuple represents a state with (SoC, state type).
                SoC values range from 0 to 100, and state types are those provided in `vehicle_states`.

        Example:
            >>> create_states(20, ["moored", "flying"])
            [(0, 'moored'), (20, 'moored'), (40, 'moored'), (60, 'moored'), (80, 'moored'), (100, 'moored'),
            (0, 'flying'), (20, 'flying'), (40, 'flying'), (60, 'flying'), (80, 'flying'), (100, 'flying')]
        """

        states = []
        for state in vehicle_states:
            for soc in range(0, 101, soc_increment):
                states.append((soc, state))
        return states
    
    def get_activation_vector(self,u,w):
        w = w[0] # Pull out the first value of w
        if w == 0:
            if u == 0:
                a = self.stm[0]
            elif u == 1:
                a = self.stm[1]
        elif w == 1:
            if u == 0:
                a = self.stm[2]
            elif u == 1:
                a = self.stm[3]
        return a

    
    def get_control_reward(self, u: str, w: list, prob):
        """
        Determines the reward for a given action.

        Args:
            daytime (int) : Indicates whether it is day (1) or night (0).
            u (char) : Describes chosen action
            condition_current (char) : Describes condition of vehicle when action is selected
            condition_next (char)
        """    
        

        if w[0] == 0: # Night time case
            if u == 'float':
                reward = 0
            elif u == 'fly':
                reward = 0
        elif w[0] == 1: # Day time case
            if u == 'float':
                reward = 0
            elif u == 'fly':
                reward = 1

        return reward

    @staticmethod
    def expected_solar_power(irradiance_mean, cloud_prob, time_of_day_factor, max_solar_power=5):
        """
        Calculates the expected solar power output for a given stage.
        
        Parameters:
            irradiance_mean (float): Mean solar irradiance (in W/m^2) at the given stage.
            cloud_prob (float): Probability of cloudiness at the stage.
            time_of_day_factor (float): A factor (0 to 1) representing the intensity of sunlight for the time of day.
            max_solar_power (float): Maximum power output of the solar system in kW (default is 5 kW).
        
        Returns:
            float: Expected solar power output in kW.
        """
        # Calculate the expected irradiance adjusted by time of day
        expected_irradiance = irradiance_mean * time_of_day_factor
        
        # Convert irradiance (W/m^2) to power in kW (assuming 1000 W/m^2 gives maximum power)
        expected_power_clear = min(max_solar_power, expected_irradiance / 1000 * max_solar_power)
        
        # Calculate the expected power output, accounting for cloudiness
        expected_power = expected_power_clear * (1 - 0.5 * cloud_prob)
        
        return expected_power
    
    @staticmethod
    def time_of_day_func(stage,timestep):
        """
        Compute the time of day as a sinusoidal function based on the current stage and timestep.

        This function models the time of day using a sine wave, where the input `stage` represents the current stage
        in the simulation, and `timestep` is the interval in minutes between each stage. The function calculates
        a normalized value between 0 and 1 representing the time of day, where the sine wave completes one full cycle
        in a 24-hour period.

        Args:
            stage (int): The current stage of the simulation, which is an integer representing the progression of time.
            timestep (int): The time interval in minutes between each stage.

        Returns:
            float: A value between 0 and 1 representing the time of day, where 0 corresponds to midnight and 1
                corresponds to the end of the day.
        """
        daily_stages = 24*60/timestep
        normalized_stage = (np.mod(stage, daily_stages) / daily_stages)
        return max(0, np.sin(np.pi * normalized_stage))

    def create_ev_table(self, max_stages):
        """
        Creates an expectation value (EV) table with a specified number of stages.

        Args:
            max_stages (int): The number of stages to define the number of columns in the EV table.

        Returns:
            None: This method does not return a value. It updates the `ev_table` attribute of the instance with a new DataFrame.

        Attributes:
            ev_table (pd.DataFrame): A DataFrame where rows correspond to states and columns correspond to stages. Each cell is initialized to `NaN`.

        """
        num_columns = max_stages

        # Create table of zeros
        self.ev_table = pd.DataFrame(
            np.nan,
            index=pd.MultiIndex.from_tuples(self.states),
            columns=range(num_columns),
        )

        for k in tqdm(range(max_stages-1,-1,-1)):
            w = [self.daytime(0,k,10)]
            for s in self.states:
                max_reward = -np.inf
                for u in self.actions:
                    control_reward = self.get_control_reward(u,w,self.prob)
                    future_reward = self.get_future_reward(s,u,k,w)
                    reward = control_reward+future_reward
                    if reward > max_reward:
                        max_reward = reward
                        # chosen_action = u
                self.ev_table.loc[s,k] = max_reward

    def calculate_soc_update(self, action: str,dt:int,stage:int)->int:
        """
        Determines update to be applied to state of charge when executing a given action at a given stage of a simulation.

        Args:
            action (str) : Action that is executed.
            dt (int)     : Time period over which the action occurs, in minutes.
            stage (int)  : Stage at which the action will be executed.

        Returns:
            int: Integer representing the change in state of charge over the given time step when the given action is taken
        """
        # Get the amount of energy collected during this timestep
        
        irradiance_mean = 1000  # W/m^2
        cloud_prob = 0.3  # Probability of cloud cover
        daily_stages = 24*60/dt
        time_of_day_factor = max(0, np.sin(np.pi * (np.mod(stage,daily_stages) / daily_stages)))

        rho = 1.2
        if action == 'float':
            u = 0
        elif action == 'flying':
            u = 20
        
        solar_power = self.expected_solar_power(irradiance_mean, cloud_prob, time_of_day_factor, max_solar_power=80)  # Watts
        required_power = self.plane.get_required_power(u,rho)  # Watts
        total_power = required_power - solar_power  # Watts
        delta_energy = total_power * dt * 60  # Joules
        
        return delta_energy/self.battery_capacity

    def expected_solar_power(irradiance_mean, cloud_prob, time_of_day_factor, max_solar_power=80):
        """
        Calculates the expected solar power output for a given stage.
        
        Parameters:
            irradiance_mean (float): Mean solar irradiance (in W/m^2) at the given stage.
            cloud_prob (float): Probability of cloudiness at the stage.
            time_of_day_factor (float): A factor (0 to 1) representing the intensity of sunlight for the time of day.
            max_solar_power (float): Maximum power output of the solar system in W (default is 80 W).
        
        Returns:
            float: Expected solar power output in kW.
        """
        # Calculate the expected irradiance adjusted by time of day
        expected_irradiance = irradiance_mean * time_of_day_factor
        
        # Convert irradiance (W/m^2) to power in kW (assuming 1000 W/m^2 gives maximum power)
        expected_power_clear = min(max_solar_power, expected_irradiance / 1000 * max_solar_power)
        
        # Calculate the expected power output, accounting for cloudiness
        expected_power = expected_power_clear * (1 - 0.5 * cloud_prob)
        
        return expected_power


    def get_future_reward(self, state, action, stage: int, w):
        """
        Function to retrieve the expected reward for a given state and stage.

        Params:
            state (tuple): State from which an action will occur
            u (int): Action to be attempted
            k (int): Stage from which action will occur

        Returns:
            reward : 
        """
        if action == "float":
            u=0
        elif action == "fly":
            u=1

        new_stage = stage+1

        if new_stage not in self.ev_table.columns:
            reward = 0
        else:
            # TODO: Determine which state the action will bring us to
            
            # Determine value of the state transition activation function
            a = self.get_activation_vector(u,w)
            soc = state[0]
            if u == 0:
                vehicle_state = self.vehicle_states[0]
            elif u == 1:
                vehicle_state = self.vehicle_states[1]
            
            new_state = (soc+a,vehicle_state)
            # Get the expected future reward for the new state
            if new_state[0] < 0 or new_state[0] > 100:
                reward = -np.inf
            else:
                reward = self.ev_table.loc[new_state, new_stage]

        return reward

    def daytime(self, start, step, dt):
        """
        Determines if the provided simulation step is during the day or at night

        Parameters:
            start (int): Time when the simulation begins (in minutes, e.g., 0 for midnight,
                         720 for noon)
            step (int) : Stage of the simulation
            dt (int) : Time, in minutes, between simulation steps

        Returns:
            int: 0 (night) or 1 (day).
        """
        # Calculate the total time passed from the start in minutes
        time_of_day = (start + step * dt) % 1440  # 1440 minutes in a day

        # Day is considered between 6 AM (360 minutes) and 6 PM (1080 minutes)
        if 360 <= time_of_day < 1080:
            return 1  # Daytime
        else:
            return 0  # Nighttime
