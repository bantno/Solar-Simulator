class DecisionProblem:
    def __init__(self, stm, rewards, sun_sequence):
        self.stm = stm
        self.rewards = rewards
        self.sun_sequence = sun_sequence  # Known sun states for each stage

    def calculate_reward(self,state,sun,action):
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
        
        if action == "flying":
            if state[1] == "flying":
                if sun == 0:
                    step_reward = self.stm[1]
                elif sun == 1:
                    step_reward = self.stm[3]
            elif state[1] == "floating":
                if sun == 0:
                    step_reward = self.stm[5]
                elif sun == 1:
                    step_reward = self.stm[7]

        if action == "floating":
            if state[1] == "flying":
                if sun == 0:
                    step_reward = self.stm[0]
                elif sun == 1:
                    step_reward = self.stm[2]
            elif state[1] == "floating":
                if sun == 0:
                    step_reward = self.stm[4]
                elif sun == 1:
                    step_reward = self.stm[6]
        
        return step_reward

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
                    possible_states.append(((next_soc, "floating"), "floating"))  # Continue floating with sun
            else:
                next_soc = soc
                possible_states.append(((next_soc, "floating"), "floating"))  # Continue floating without sun (no change in SoC)

        # Include the option to switch modes at each step, ensuring valid SoC
        if mode == "flying":
            if sun == 1:
                next_soc = soc + 20
                if next_soc <= 100:
                    possible_states.append(((next_soc, "floating"), "floating"))  # Switch to floating with sun
            else:
                next_soc = soc
                possible_states.append(((next_soc, "floating"), "floating"))  # Switch to floating without sun
        else:
            if sun == 1:
                next_soc = soc - 20
                if next_soc >= 0:
                    possible_states.append(((next_soc, "flying"), "flying"))  # Switch to flying with sun
            else:
                next_soc = soc - 40
                if next_soc >= 0:
                    possible_states.append(((next_soc, "flying"), "flying"))  # Switch to flying without sun

        return possible_states

    def backward_induction(self, stage, state, stages_remaining):
        """
        Performs backward induction to calculate the optimal reward and the sequence of states and actions.

        Parameters:
        stage (int): The current stage in the decision process.
        state (tuple): The current state, (SoC, "flying" or "floating").
        stages_remaining (int): The number of stages remaining.

        Returns:
        tuple: A tuple containing the optimal reward and the corresponding sequence of states, actions, and cumulative rewards.
        """
        if stages_remaining == 0:
            # Base case: No more stages remaining
            return 0, [(stage, state, None, 0)]  # No action, no cumulative reward

        # Get the known sun state for this stage
        sun = self.sun_sequence[stage]

        # Recursive case: Explore all possible next states
        max_reward = float('-inf')
        best_path = []

        for next_state, action in self.get_possible_next_states(state, sun):
            next_reward, next_path = self.backward_induction(stage + 1, next_state, stages_remaining - 1)
            reward = self.calculate_reward(state, sun, action) + next_reward

            # Prefer maintaining the same mode if there's a tie in expected reward
            if (reward > max_reward) or (reward == max_reward and action == state[1]):
                max_reward = reward
                cumulative_reward = reward
                best_path = [(stage, state, action, cumulative_reward)] + next_path
        
        return max_reward, best_path



# Example usage:
# stm: decision thresholds
# rewards: corresponding rewards
stm = [0, -40, 20, -20, 0, -40, 20, -20]
rewards = [0, 0, 0, 10, 0, 0, 0, 0]

# Known sun states for each stage
day = [0,0,1,1,1,1,1,1,1,1,1,1,1,1,0,0]
day = [1,1,0,0]
sun_sequence = [1, 1, 0]  # Example sun sequence over 3 stages
sun_sequence = day

# Initial state: (State of Charge (SoC), "flying" or "floating")
initial_state = (40, "flying")

# Number of stages to consider
stages_remaining = len(sun_sequence) - 1  # Adjusting to simulate the correct number of stages

problem = DecisionProblem(stm, rewards, sun_sequence)

# Start backward induction from the first stage (stage 0)
optimal_reward, optimal_path = problem.backward_induction(0, initial_state, stages_remaining)
print(f"Optimal reward: {optimal_reward}")
print("Optimal path:")
for step in optimal_path:
    print(step)
