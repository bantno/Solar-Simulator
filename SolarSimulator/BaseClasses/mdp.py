import pandas as pd
import numpy as np


class mdp:
    """
    Class representing a markov decision process problem
    """
    def __init__(self,states,max_stages,stm,reward):
        self.states=states
        self.stm = stm
        self.reward = reward
        self.reward_table = np.full((len(self.states),max_stages),np.nan)

    def create_table(self,charges, y_states, actions, transitions, rewards):
        """
        Create state transition table
        """
        data = []
        for charge in charges:
            for y in y_states:
                for sun in [0, 1]:
                    for action, transition, reward in zip(actions, transitions[sun], rewards[sun]):
                        next_charge = transition  # Ensure charge stays within 0-100
                        data.append([charge, y, sun, action, next_charge, reward])
        
        df = pd.DataFrame(data, columns=["x (State of Charge) @ t=i", "y @ t=i", "Sun?", "Action @ t=i", "delta_x @ t=i+1", "Reward @ i"])
        return df

    
    def calculate_reward(self,state,action,sun):
        """
        Calculates the current reward based on the state, action, and sun.

        Parameters:
        state (tuple): The current state, (SoC, "flying" or "floating")
        sun (int): The sun state, 0 or 1.
        stm (list):
        reward (list):

        Returns:
        int: The reward based on the provided table.
        """
        
        if action == 1:
            if state[1] == "flying":
                if sun == 0:
                    step_reward = self.stm[1]
                elif sun == 1:
                    step_reward = self.stm[3]
            elif state[1] == "moored":
                if sun == 0:
                    step_reward = self.stm[5]
                elif sun == 1:
                    step_reward = self.stm[7]

        if action == 0:
            if state[1] == "flying":
                if sun == 0:
                    step_reward = self.stm[0]
                elif sun == 1:
                    step_reward = self.stm[2]
            elif state[1] == "moored":
                if sun == 0:
                    step_reward = self.stm[4]
                elif sun == 1:
                    step_reward = self.stm[6]
            
                

        # If the inputs do not match any of the conditions in the table, return None or raise an error.
        return step_reward
    
    def daylight(self,hour):
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
        
    def calculate_table(self):
        entry = []
        for k in range(MAX_STAGES-1,-1,-1):
            for i,state in enumerate(self.states):
                # TODO: Make this a function
                # Entry is simply single step reward if filling out last column of 
                if k==MAX_STAGES-1:
                    entry = self.calculate_reward(state,self.daylight(k))
                    self.reward_table[i,k] = entry
                else:
                    entry = self.calculate_reward

        
        return None




# Example usage
states=[]
for state in ["moored","flying"]:
    for soc in range(0,101,20):
        states.append((soc,state))
print(states)

stm = [0,-40,20,-20,0,-40,20,-20]
reward = [0,0,0,10,0,0,0,0]
MAX_STAGES=20
mdproblem = mdp(states,MAX_STAGES,stm,reward)
mdproblem.calculate_table()
print(mdproblem.reward_table)

