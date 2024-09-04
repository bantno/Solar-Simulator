import pandas as pd
import numpy as np


class mdp:
    """
    Class representing a markov decision process problem
    """

    def __init__(self, soc_increment, vehicle_states, max_stages, actions, weights):

        self.states = self.create_states(soc_increment, vehicle_states)
        self.actions = actions
        self.w = weights
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
    
    @staticmethod
    def get_activation_vector(u):
        pass

    @staticmethod
    def get_control_reward(u: str, w: list):
        """
        Computes a control reward based on the input action string `u` and a list of weights `w`.

        Args:
            u (str): A string representing the control action. Expected values are:
                - 'float': Represents a floating action, which is internally mapped to 0.
                - 'fly': Represents a flying action, which is internally mapped to 1.
            w (list): A list of numerical values (weights) to be scaled by the action.

        Returns:
            reward (float): A scalar reward calculated by sequentially multiplying the action value
                   (`0` for 'float' or `1` for 'fly') with each element in the list `w`.
        """

        if u == "float":
            u = 0
        elif u == "fly":
            u = 1

        reward = u
        for element in w:
            reward *= element

        return reward

    def create_ev_table(self, max_stages):
        """
        Creates an empty expectation value (EV) table with a specified number of stages.

        Args:
            max_stages (int): The number of stages to define the number of columns in the EV table.

        Returns:
            None: This method does not return a value. It updates the `ev_table` attribute of the instance with a new DataFrame.

        Attributes:
            ev_table (pd.DataFrame): A DataFrame where rows correspond to states and columns correspond to stages. Each cell is initialized to `NaN`.

        Examples:
            If `self.states` is `['state1', 'state2']` and `max_stages` is 3, the resulting `ev_table` will look like:

            ```
                   0   1   2
            state1  NaN NaN NaN
            state2  NaN NaN NaN
            ```
        """
        num_columns = max_stages

        # Create table of zeros
        self.ev_table = pd.DataFrame(
            np.nan,
            index=self.states,
            columns=range(num_columns),
        )

        for k in range(max_stages,-1,-1):
            for s in self.states:
                max_reward = -np.inf
                for u in self.actions:
                    control_reward = self.get_control_reward(u,self.w)
                    future_reward = self.get_future_reward(s,u,k)
                    reward = control_reward+future_reward
                    if reward > max_reward:
                        max_reward = reward
                        # chosen_action = u
                self.ev_table[s,k] = max_reward

    def get_future_reward(self, state, action, stage: int):
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
            a = self.get_activation_vector(u)
            new_state = state + a
            # Get the expected future reward for the new state
            reward = self.ev_table.loc[new_state, new_stage]

        return reward
    


    def get_value(self, state, u, w, k):
        """
        Function to determine the value of being in a given state at a given stage.

        Params:
            i (int): Index of the list of states that describes the current state
            k (int): Current stage (typically timestep)

        Returns:
            value (float):
                Value of being in the provided state at the provided stage.

        """
        control_reward = self.get_control_reward(u, w)
        future_reward = self.get_future_reward(state, u, k)
        return control_reward + future_reward

    def daylight(self, hour):
        """
        Returns 0 if the input hour modulo 24 is between 0 and 5 or 18 and 23,
        and 1 otherwise.

        Parameters:
        hour (int): The input hour.

        Returns:
        int: 0 or 1 based on the conditions.
        """
        hour_mod = hour % 24
        if 0 <= hour_mod <= 5 or 18 <= hour_mod <= 23:
            return 0
        else:
            return 1
