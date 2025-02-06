from stable_baselines3 import PPO, DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
import numpy as np

class SimpleMarketMaker:
    def __init__(self, env, bid_spread_fraction=0.5, ask_spread_fraction=0.5):
        """
        Initialize the market-making agent.

        Parameters:
        - bid_spread_fraction: Fraction of the spread to place the bid below the midpoint.
        - ask_spread_fraction: Fraction of the spread to place the ask above the midpoint.
        """
        self.bid_spread_fraction = bid_spread_fraction
        self.ask_spread_fraction = ask_spread_fraction

    def act(self, observation):
        """
        Decide the bid and ask prices based on the observation.

        Parameters:
        - observation: A list or array containing the environment's observation.
          The first element is expected to be the midpoint, and the second the spread.

        Returns:
        - action: A tuple (bid_price, ask_price).
        """
        midpoint = observation[0]  # Extract the midpoint
        spread = observation[1]    # Extract the spread
        bid_price = midpoint - (spread * self.bid_spread_fraction)
        ask_price = midpoint + (spread * self.ask_spread_fraction)
        return bid_price, ask_price
    
class PPOMarketMaker:
    def __init__(self, env):
        train_env = DummyVecEnv([lambda: env])
        self.model = PPO("MlpPolicy", train_env, learning_rate=0.01)
        self.model.learn(total_timesteps=100_000)
        env.reset()
    
    def act(self, observation):
        action, _ = self.model.predict(observation)
        return action
    
class DQNMarketMaker:
    def __init__(self, env):
        train_env = make_vec_env(lambda: env, n_envs=1)
        
        self.model = DQN(
            policy="MlpPolicy",           # Use a multi-layer perceptron policy
            env=train_env,                # Your custom environment
            learning_rate=1e-3,           # Learning rate
            buffer_size=10000,            # Replay buffer size
            learning_starts=1000,         # Steps before training begins
            batch_size=64,                # Batch size for training
            tau=0.1,                      # Target network update rate
            gamma=0.99,                   # Discount factor
            train_freq=4,                 # Train every 4 steps
            target_update_interval=500,   # Update target network every 500 steps
        )
        self.model.learn(total_timesteps=100_000)
        env.reset()
        
    def act(self, observation):
        action, _ = self.model.predict(observation, deterministic=True)
        return action
    
class ASMarketMaker:
    def __init__(self, env, gamma = 0.1, sigma = 0, kappa = 1.5, T = 0):
        self.gamma = gamma
        #As we hypothese that there is no closure time, sigma won't be used
        self.sigma = sigma
        self.kappa = kappa
        #And T either
        self.T = T
    def compute_spread(self, inventory):
        #We make the hypothese that there is no closure time
        ret = (2 / self.gamma) * np.log(1 + self.gamma * inventory / self.kappa)
        return ret
    def act(self, observation):
        midpoint = observation[0]
        inventory = observation[2]
        spread = self.compute_spread(inventory)
        bid_price = midpoint - spread / 2
        ask_price = midpoint + spread / 2
        return bid_price, ask_price