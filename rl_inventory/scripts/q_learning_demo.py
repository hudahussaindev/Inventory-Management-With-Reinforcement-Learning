from rl_inventory.envs.extended_inventory import ExtendedInventoryEnv
from rl_inventory.agents.qlearning.qlearning import QLearningAgent, StateDiscretizer

import numpy as np

def train_agent(
    num_episodes=200,
    max_steps_per_episode=365,
):
    env = ExtendedInventoryEnv()
    disc = StateDiscretizer()
    agent = QLearningAgent(
        n_actions=env.action_space.n,
        alpha=0.1,
        gamma=0.99,
        epsilon=0.2
    )

    for episode in range(num_episodes):
        state = env.reset()
        s = disc.discretize(state)
        total_reward = 0

        for _ in range(max_steps_per_episode):
            a = agent.select_action(s)
            next_state, reward, terminated, truncated, info = env.step(a)
            s_next = disc.discretize(next_state)
            agent.update(s, a, reward, s_next, terminated)
            s = s_next
            total_reward += reward
            if terminated or truncated:
                break

        if (episode + 1) % 20 == 0:
            print(f"Episode {episode+1}, total reward = {total_reward:.2f}")

    return agent, disc


def demo_run(agent, disc):
    env = ExtendedInventoryEnv()
    state = env.reset()
    s = disc.discretize(state)
    total_reward = 0

    for day in range(env.episode_length):
        q_vals = [agent.Q[(s, a)] for a in range(agent.n_actions)]
        a = int(np.argmax(q_vals))

        next_state, reward, terminated, truncated, info = env.step(a)
        total_reward += reward

        print(
            f"Day={day:03d}  Inv={info['inventory']}  "
            f"Ord={info['order_quantity']}  Demand={info['demand']}  "
            f"Lost={info['lost_sales']}  Cost={info['total_cost']:.2f}"
        )

        env.render()
        s = disc.discretize(next_state)
        if terminated or truncated:
            break

    print(f"Demo episode finished. Total reward = {total_reward:.2f}")
    env.plot_episode()


def main():
    agent, disc = train_agent(num_episodes=200)
    demo_run(agent, disc)


if __name__ == "__main__":
    main()
