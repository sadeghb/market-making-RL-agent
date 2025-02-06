# Imports
import os
import kaggle
import pandas as pd
from stable_baselines3.common.env_checker import check_env

from helpers import plot_results, test_market_maker_agent
from environments import SimpleMarketMakingEnv, PPOMarketMakingEnv, DQNMarketMakingEnv, ASMarketMakingEnv, RewardType
from agents import SimpleMarketMaker, PPOMarketMaker, DQNMarketMaker, ASMarketMaker


# -------------------------------------------------------------------------------------------------------------------------------------
# FETCHING DATA

file = "./data/BTC_1sec.csv"

if not os.path.exists(file):
    # if this line gives you an error, you need to authenticate with kaggle (https://www.kaggle.com/docs/api -- Authentication section)
    kaggle.api.authenticate()
    kaggle.api.dataset_download_files('martinsn/high-frequency-crypto-limit-order-book-data', path='./data/', unzip=True)
    
dataset = pd.read_csv(file).iloc[:, 1:]
length = 200_000
print(f"dataset of length {length}")

# -------------------------------------------------------------------------------------------------------------------------------------
# Evaluating

def evaluate(env_class, agent_class, label, reward_type: RewardType):
    env = env_class(dataset, initial_cash=5_500_000, initial_inventory=100, trade_volume=1, inventory_penalty=0.001, reward_type=reward_type)
    check_env(env)
    
    # this might take some time because some models might be pre-training
    agent = agent_class(env)
    
    # Run the test
    results = test_market_maker_agent(env, agent, steps=length)
    plot_results([results], [label])
    return results
    
# Reward 1
simple_results_1 = evaluate(SimpleMarketMakingEnv, SimpleMarketMaker, "simple", RewardType.REWARD1)
ppo_results_1 = evaluate(PPOMarketMakingEnv, PPOMarketMaker, "ppo", RewardType.REWARD1)
dqn_results_1 = evaluate(DQNMarketMakingEnv, DQNMarketMaker, "dqn", RewardType.REWARD1)
as_results_1 = evaluate(ASMarketMakingEnv, ASMarketMaker, "Avellaneda Stoikov", RewardType.REWARD1)

plot_results([simple_results_1, as_results_1, ppo_results_1, dqn_results_1], ["simple", "a-s", "ppo", "dqn"])

# Reward 2
simple_results_2 = evaluate(SimpleMarketMakingEnv, SimpleMarketMaker, "simple", RewardType.REWARD2)
ppo_results_2 = evaluate(PPOMarketMakingEnv, PPOMarketMaker, "ppo", RewardType.REWARD2)
dqn_results_2 = evaluate(DQNMarketMakingEnv, DQNMarketMaker, "dqn", RewardType.REWARD2)
as_results_2 = evaluate(ASMarketMakingEnv, ASMarketMaker, "Avellaneda Stoikov", RewardType.REWARD2)

plot_results([simple_results_2, as_results_2, ppo_results_2, dqn_results_2], ["simple", "a-s", "ppo", "dqn"])

# Final Comparison
plot_results([as_results_1, simple_results_1, ppo_results_1, ppo_results_2], ["a-s", "simple", "ppo reward 1", "ppo reward 2"], no_reward=True)