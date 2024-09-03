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
            
        return step_reward
    
    def get_future_reward(self,state,stage):
        """Function that retrieves expected reward for the given state and stage"""
        i=0
        for s in self.states:
            if state == s:
                break
            else:
                i+=1
        
        return self.reward_table[i,stage]
    
    def get_possible_next_states(self, state, sun):
        """
        Generates possible next states based on the current state and sun.

        Parameters:
        state (tuple): The current state, (SoC, "flying" or "floating").
        sun (int): The sun state, 0 or 1.

        Returns:
        list of tuples: Possible next states, each paired with the action taken.
        """
        soc, mode = state
        possible_states = []

        if mode == "flying":
            if sun == 1:
                next_soc = soc - 20
                if next_soc >= 0:
                    possible_states.append(((next_soc, "flying"), "flying"))  # Continue flying with sun
            else:
                next_soc = soc - 40
                if next_soc >= 0:
                    possible_states.append(((next_soc, "flying"), "flying"))  # Continue flying without sun

        elif mode == "floating":
            if sun == 1:
                next_soc = soc + 20
                if next_soc <= 100:
                    possible_states.append((next_soc, "floating"))  # Continue floating with sun
            else:
                next_soc = soc
                possible_states.append((next_soc, "floating"))  # Continue floating without sun (no change in SoC)

        # Include the option to switch modes at each step, ensuring valid SoC
        if mode == "flying":
            if sun == 1:
                next_soc = soc + 20
                if next_soc <= 100:
                    possible_states.append((next_soc, "floating"))  # Switch to floating with sun
            else:
                next_soc = soc
                possible_states.append((next_soc, "floating"))  # Switch to floating without sun
        else:
            if sun == 1:
                next_soc = soc - 20
                if next_soc >= 0:
                    possible_states.append((next_soc, "flying"))  # Switch to flying with sun
            else:
                next_soc = soc - 40
                if next_soc >= 0:
                    possible_states.append((next_soc, "flying"))  # Switch to flying without sun

        return possible_states
    
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

        # for k in range(MAX_STAGES-1,-1,-1):
        #     sun = self.daylight(k)
        #     for i,state in enumerate(self.states):
        #         # TODO: Make this a function

        #         if k==MAX_STAGES-1:
        #             entry = self.calculate_reward(state,0,self.daylight(k)) # need to figure out how to do the terminal calculation
        #             self.reward_table[i,k] = entry
        #         else:
        #             candidates = []
        #             for next_state in self.get_possible_next_states(state,sun):
        #                 action = 0
        #                 candidate = self.calculate_reward(state,action,sun) + self.get_future_reward(next_state,k+1)

        
        return None




# Example usage
states=[]
for state in ["moored","flying"]:
    for soc in range(0,101,20):
        states.append((soc,state))
# print(states)

stm = [0,-40,20,-20,0,-40,20,-20]
reward = [0,0,0,10,0,0,0,0]
MAX_STAGES=20
mdproblem = mdp(states,MAX_STAGES,stm,reward)
initial = (40,"flying")
print(mdproblem.get_possible_next_states(state=initial,sun=1))


# mdproblem.calculate_table()
# print(mdproblem.reward_table)

