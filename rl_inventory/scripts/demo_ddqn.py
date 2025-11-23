"""Demo script for training and testing Double DQN agent."""

# Path fix MUST be first, before any rl_inventory imports
from rl_inventory.agents.DoubleDQN.DoubleDQN import DoubleDQNAgent
from rl_inventory.envs.ExtendedInventoryEnv_DDQN import ExtendedInventoryEnv_DDQN
import numpy as np


def train_ddqn(
    num_episodes=1000,
    max_steps_per_episode=365,
    lr=0.001,
    gamma=0.99,
    epsilon=1.0,
    epsilon_min=0.01,
    epsilon_decay=0.995,
    batch_size=64,
    buffer_size=10000,
    target_update_freq=100,
    hidden_size=128
):
    # Initialize environment
    env = ExtendedInventoryEnv_DDQN(
        discrete_actions=True,
        seed=42,
        reward_scale=100.0,
        use_reward_shaping=False
    )

    # Initialize agent
    agent = DoubleDQNAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=9,
        lr=lr,
        gamma=gamma,
        epsilon=epsilon,
        epsilon_min=epsilon_min,
        epsilon_decay=epsilon_decay,
        batch_size=batch_size,
        buffer_size=buffer_size,
        target_update_freq=target_update_freq,
        hidden_size=hidden_size
    )

    # Training metrics
    episode_rewards = []
    episode_costs = []
    episode_losses = []

    print("Training Double DQN agent...")
    print(f"State dimension: {env.observation_space.shape[0]}")
    print(f"Action dimension: {env.action_space.n}")
    print("-" * 60)

    # Training loop
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        total_cost = 0
        episode_loss_sum = 0
        loss_count = 0

        for step in range(max_steps_per_episode):
            # Select action
            action = agent.select_action(state, training=True)

            # Take step
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Store transition
            agent.store_transition(state, action, reward, next_state, done)

            # Train and track loss
            loss = agent.train_step()
            if loss is not None:
                episode_loss_sum += loss
                loss_count += 1

            # Update metrics
            total_reward += reward
            total_cost += info.get('total_cost', -reward * env.reward_scale)
            state = next_state

            if done:
                break

        # Decay epsilon
        agent.decay_epsilon()

        # Record episode metrics
        episode_rewards.append(total_reward)
        episode_costs.append(total_cost)
        avg_episode_loss = episode_loss_sum / loss_count if loss_count > 0 else None
        episode_losses.append(avg_episode_loss)

        # Print progress every 100 episodes
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            avg_cost = np.mean(episode_costs[-100:])
            recent_losses = [l for l in episode_losses[-100:] if l is not None]
            avg_loss = np.mean(recent_losses) if recent_losses else None

            loss_str = f"Avg Loss: {avg_loss:.4f}" if avg_loss is not None else "Avg Loss: N/A"
            print(f"Episode {episode+1}/{num_episodes} | "
                  f"Avg Reward: {avg_reward:.2f} | "
                  f"Avg Cost: {avg_cost:.2f} | "
                  f"{loss_str} | "
                  f"Epsilon: {agent.epsilon:.3f}")

    print("\nTraining completed!")
    return agent, env


def test_ddqn(agent, env, num_episodes=10, seed=42):
    """Test the trained agent."""
    env.seed(seed)  # seeded
    print(f"\nTesting agent over {num_episodes} episodes...")

    episode_costs = []

    for episode in range(num_episodes):
        state = env.reset()
        total_cost = 0

        for step in range(env.episode_length):
            action = agent.predict(state)  # Greedy action
            next_state, reward, terminated, truncated, info = env.step(action)
            total_cost += info.get('total_cost', -reward * env.reward_scale)
            state = next_state

            if terminated or truncated:
                break

        episode_costs.append(total_cost)
        print(f"Episode {episode+1}: Total Cost = ${total_cost:.2f}")

    avg_cost = np.mean(episode_costs)
    std_cost = np.std(episode_costs)
    print(f"\nAverage Cost: ${avg_cost:.2f} (±${std_cost:.2f})")

    return episode_costs


def demo_run(agent, env, seed=42):
    """Run a single demo episode with detailed output and charts."""
    env.seed(seed)  # seeded
    print("\n" + "=" * 60)
    print("Running Demo Episode (Greedy Policy)")
    print("=" * 60)

    state = env.reset()
    total_reward = 0
    total_cost = 0

    for day in range(env.episode_length):
        # Get action (no exploration)
        action = agent.predict(state)

        # Take step
        next_state, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        total_cost += info.get('total_cost', -reward * env.reward_scale)

        # Print day-by-day information
        print(
            f"Day={day:03d}  Inv={info['inventory']:6.0f}  "
            f"Ord={info['order_quantity']:6.0f}  Demand={info['demand']:6.0f}  "
            f"Lost={info['lost_sales']:6.0f}  Cost={info['total_cost']:8.2f}"
        )

        state = next_state

        if terminated or truncated:
            break

    print(f"\nDemo episode finished.")
    print(f"Total Reward: {total_reward:.2f}")
    print(f"Total Cost: ${total_cost:.2f}")

    # Plot the episode charts
    env.plot_episode()


if __name__ == "__main__":
    # Train agent
    agent, env = train_ddqn(num_episodes=1000)

    # Test agent
    test_ddqn(agent, env, num_episodes=10)

    # Run demo episode
    demo_run(agent, env)
