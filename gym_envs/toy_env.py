import gym
import numpy as np
from gym import spaces


class ToyOfflineEnv(gym.Env):
    """
    Custom Toy Offline Environment for loading predefined trajectories.
    """
    def __init__(self, trajectories):
        super(ToyOfflineEnv, self).__init__()

        # Store trajectories
        self.trajectories = trajectories
        self.current_episode = 0
        self.current_step = 0

        # Define observation & action space (adjust as needed)
        self.observation_space = spaces.Discrete(10)  # Example: 10 possible states
        self.action_space = spaces.Discrete(2)  # Example: 2 possible actions (Pass, Bet)

        # Current state info
        self.state = None

    def reset(self):
        """
        Reset the environment to the start of a trajectory.
        """
        self.current_step = 0
        self.current_episode = np.random.randint(0, len(self.trajectories))  # Random episode selection
        self.state = self.trajectories[self.current_episode]["num_states"][self.current_step]

        return self.state  # Returning state

    def step(self, action):
        """
        Step through the trajectory.
        """
        episode = self.trajectories[self.current_episode]

        # Check if we've reached the last step
        if self.current_step >= len(episode["num_states"]) - 1:
            done = True
            reward = sum(episode["rewards"])  # Sum up all rewards
            return self.state, reward, done, {}
        
        # Move to next step
        self.current_step += 1
        self.state = episode["num_states"][self.current_step]

        # Get the reward (if available for this step)
        reward = episode["rewards"][self.current_step - 1] if self.current_step - 1 < len(episode["rewards"]) else 0
        done = False

        return self.state, reward, done, {}

    def render(self, mode="human"):
        """
        Render environment state.
        """
        print(f"Current State: {self.state}")