import matplotlib.pyplot as plt

def test_market_maker_agent(env, agent, steps=10):
    """
    Test the market-making agent in the provided environment.

    Parameters:
    - env: The market-making environment.
    - agent: The market-making agent.
    - steps: Number of steps to simulate.

    Returns:
    - results: A dictionary containing cash, inventory, rewards, and midpoint over time.
    """
    obs, _ = env.reset()
    cash_list = [env.cash]
    inventory_list = [env.inventory]
    reward_list = []
    cumulative_reward_list = []
    midpoint_list = [obs[0]]  # Store the midpoint
    wealth = []

    cumulative_reward = 0

    for _ in range(steps):
        action = agent.act(obs)
        obs, reward, done, _, _ = env.step(action)
        cash_list.append(env.cash)
        inventory_list.append(env.inventory)
        reward_list.append(reward)
        cumulative_reward += reward
        cumulative_reward_list.append(cumulative_reward)
        midpoint_list.append(obs[0])
        wealth.append(env.cash + obs[0] * env.inventory)

        if done:
            env.reset()

    return {
        "cash": cash_list,
        "inventory": inventory_list,
        "rewards": reward_list,
        "cumulative_rewards": cumulative_reward_list,
        "midpoint": midpoint_list,
        "wealth": wealth
    }
    

def plot_results(results, labels, no_reward=False):
    # Plot results
    plt.figure(figsize=(14, 10))
    
    assert len(results) == len(labels)

    plt.subplot(4, 1, 1)
    for r, la in zip(results, labels):
        plt.plot(r['cash'], marker=',', label=la)
    plt.title("Agent's Cash Over Time")
    plt.ylabel("Cash ($)")
    plt.grid(True)
    plt.legend()

    # Inventory over time
    plt.subplot(4, 1, 2)
    for r, la in zip(results, labels):
        plt.plot(r['inventory'], marker=',', label=la)
    plt.title("Agent's Inventory Over Time")
    plt.ylabel("Inventory (units)")
    plt.grid(True)
    plt.legend()

    next = 3
    if not no_reward:
        # Cumulative reward over time
        plt.subplot(4, 1, 3)
        for r, la in zip(results, labels):
            plt.plot(r['cumulative_rewards'], marker=',', label=la)
        plt.title("Agent's Cumulative Reward Over Time")
        plt.xlabel("Timestep")
        plt.ylabel("Reward ($)")
        plt.grid(True)
        plt.legend()
        next += 1

    # Midpoint price over time
    start_wealth = results[0]["wealth"][0]
    baseline = [start_wealth] * len(results[0]["wealth"])
    
    plt.subplot(4, 1, next)
    for r, la in zip(results, labels):
        plt.plot(r['wealth'], marker=',', label=la)
    plt.plot(baseline, marker=',', label="initial wealth")
    plt.title("Wealth Over Time")
    plt.xlabel("Timestep")
    plt.ylabel("Wealth ($)")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()