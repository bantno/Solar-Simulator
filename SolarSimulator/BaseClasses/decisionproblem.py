class DecisionProblem:
    def __init__(self, stm, rewards, sun_sequence):
        self.stm = stm
        self.rewards = rewards
        self.sun_sequence = sun_sequence  # Known sun states for each stage

    def calculate_reward(self, state, sun, prev_reward):
        """
        Calculates the reward based on the state, sun, and action.

        Parameters:
        state (tuple): The current state, (SoC, "flying" or "floating").
        sun (int): The sun state, 0 or 1.
        prev_reward (int): The cumulative reward from the previous stages.

        Returns:
        int: The cumulative reward after adding the step reward.
        """
        
        if state[1] == "flying":
            if sun == 0:
                if state[0] >= self.stm[1]:  # Choose to fly
                    step_reward = self.rewards[1]
                else:
                    step_reward = self.rewards[0]
            elif sun == 1:
                if state[0] >= self.stm[3]:
                    step_reward = self.rewards[3]
                else:
                    step_reward = self.rewards[2]
                
        elif state[1] == "floating":
            if sun == 0:
                if state[0] >= self.stm[5]:
                    step_reward = self.rewards[5]
                else:
                    step_reward = self.rewards[4]
            elif sun == 1:
                if state[0] >= self.stm[7]:
                    step_reward = self.rewards[7]
                else:
                    step_reward = self.rewards[6]

        return step_reward + prev_reward

    def get_possible_next_states(self, state, sun):
        """
        Generates possible next states based on the current state and sun.

        Parameters:
        state (tuple): The current state, (SoC, "flying" or "floating").
        sun (int): The sun state, 0 or 1.

        Returns:
        list of tuples: Possible next states.
        """
        soc, mode = state
        possible_states = []

        if mode == "flying":
            if sun == 1:
                next_soc = soc - 20
                if next_soc >= 0:
                    possible_states.append((next_soc, "flying"))  # Continue flying with sun
            else:
                next_soc = soc - 40
                if next_soc >= 0:
                    possible_states.append((next_soc, "flying"))  # Continue flying without sun

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

    def backward_induction(self, stage, state, stages_remaining):
        """
        Performs backward induction to calculate the optimal reward and the sequence of states and actions.

        Parameters:
        stage (int): The current stage in the decision process.
        state (tuple): The current state, (SoC, "flying" or "floating").
        stages_remaining (int): The number of stages remaining.

        Returns:
        tuple: A tuple containing the optimal reward and the corresponding sequence of states and actions.
        """
        if stages_remaining == 0:
            # Base case: No more stages remaining
            return 0, [(stage, state)]

        # Get the known sun state for this stage
        sun = self.sun_sequence[stage]

        # Recursive case: Explore all possible next states
        max_reward = float('-inf')
        best_path = []

        for next_state in self.get_possible_next_states(state, sun):
            next_reward, next_path = self.backward_induction(stage + 1, next_state, stages_remaining - 1)
            reward = self.calculate_reward(state, sun, next_reward)

            if reward > max_reward:
                max_reward = reward
                best_path = [(stage, state)] + next_path
        
        return max_reward, best_path


# Example usage:
# stm: decision thresholds
# rewards: corresponding rewards
stm = [0, -40, 20, -20, 0, -40, 20, -20]
rewards = [0, 0, 0, 10, 0, 0, 0, 0]

# Known sun states for each stage
sun_sequence = [1, 1, 0, 0]  # Example sun sequence over 3 stages

# Initial state: (State of Charge (SoC), "flying" or "floating")
initial_state = (20, "flying")

# Number of stages to consider
stages_remaining = len(sun_sequence)-1

problem = DecisionProblem(stm, rewards, sun_sequence)

# Start backward induction from the first stage (stage 0)
optimal_reward, optimal_path = problem.backward_induction(0, initial_state, stages_remaining)
print(f"Optimal reward: {optimal_reward}")
print("Optimal path:")
for step in optimal_path:
    print(step)
