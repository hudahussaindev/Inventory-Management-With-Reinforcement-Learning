"Evaluation Framework for Inventory Management RL"

import os
import pickle
from datetime import datetime
from typing import Callable, Dict, List, Optional, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from rl_inventory.agents.DoubleDQN.DoubleDQN import DoubleDQNAgent
from rl_inventory.envs.extended_inventory import ExtendedInventoryEnv
from rl_inventory.envs.extended_inventory_ppo import ExtendedInventoryEnvPPO
from rl_inventory.envs.extended_inventory_sac import ExtendedInventoryEnvSAC
from rl_inventory.envs.ExtendedInventoryEnv_DDQN import ExtendedInventoryEnv_DDQN
from rl_inventory.agents.qlearning.qlearning import QLearningAgent, StateDiscretizer
from rl_inventory.scripts.q_learning_demo import train_agent as train_q_agent
from rl_inventory.scripts.ppo_demo import train_agent as train_ppo_agent
from rl_inventory.scripts.sac_demo import train_agent as train_sac_agent
from rl_inventory.scripts.dyna_q_demo import train_agent as train_dyna_agent
from rl_inventory.scripts.demo_ddqn import train_ddqn


EnvFactory = Callable[[Optional[int]], ExtendedInventoryEnv]

# Base path for saving models (relative to project root)
BASE_AGENTS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "agents")


def get_model_save_path(agent_name: str) -> str:
    "Get the path to save/load models for a specific agent."
    path = os.path.join(BASE_AGENTS_PATH, agent_name, "trained_models")
    os.makedirs(path, exist_ok=True)
    return path


def generate_model_name() -> str:
    "Generate a unique model name based on timestamp."
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def list_saved_models(agent_name: str) -> List[str]:
    "List all saved models for a given agent."
    path = get_model_save_path(agent_name)
    if not os.path.exists(path):
        return []

    models = []
    for item in os.listdir(path):
        item_path = os.path.join(path, item)
        if os.path.isfile(item_path) or os.path.isdir(item_path):
            # Remove extension for display
            name = os.path.splitext(
                item)[0] if os.path.isfile(item_path) else item
            if name not in models:
                models.append(name)
    return sorted(models)


def prompt_model_name(agent_name: str) -> str:
    "Prompt user to generate or enter a custom model name."
    print(f"\nModel naming for {agent_name}:")
    print("  [1] Auto-generate name")
    print("  [2] Enter custom name")

    while True:
        choice = input("Choice (1/2): ").strip()
        if choice == "1":
            name = generate_model_name()
            print(f"  Generated name: {name}")
            return name
        elif choice == "2":
            name = input("  Enter model name: ").strip()
            if name:
                return name
            print("  Name cannot be empty.")
        else:
            print("  Invalid choice. Enter 1 or 2.")


def prompt_train_or_load(agent_name: str) -> tuple[bool, Optional[str]]:
    "Prompt user to train a new model or load an existing one."
    saved_models = list_saved_models(agent_name)

    print(f"\n{'='*50}")
    print(f" {agent_name.upper()} ")
    print(f"{'='*50}")

    if saved_models:
        print(f"\nFound {len(saved_models)} saved model(s):")
        for i, model in enumerate(saved_models, 1):
            print(f"  [{i}] {model}")
        print(f"  [N] Train new model")

        while True:
            choice = input(
                f"\nSelect model to load or 'N' to train new: ").strip().upper()
            if choice == "N":
                model_name = prompt_model_name(agent_name)
                return True, model_name
            try:
                idx = int(choice) - 1
                if 0 <= idx < len(saved_models):
                    return False, saved_models[idx]
                print(
                    f"  Invalid selection. Enter 1-{len(saved_models)} or 'N'.")
            except ValueError:
                print(f"  Invalid input. Enter a number or 'N'.")
    else:
        print("\nNo saved models found. Training new model...")
        model_name = prompt_model_name(agent_name)
        return True, model_name


# Model Save/Load Functions

def save_qlearning_model(agent: QLearningAgent, discretizer: StateDiscretizer,
                         agent_name: str, model_name: str) -> str:
    """Save Q-Learning or Dyna-Q agent (Q-table + discretizer)."""
    path = get_model_save_path(agent_name)
    filepath = os.path.join(path, f"{model_name}.pkl")

    save_data = {
        "Q": dict(agent.Q),
        "n_actions": agent.n_actions,
        "discretizer_bins": discretizer.bins,
    }

    with open(filepath, "wb") as f:
        pickle.dump(save_data, f)

    print(f"  Saved model to: {filepath}")
    return filepath


def load_qlearning_model(agent_name: str, model_name: str) -> tuple[QLearningAgent, StateDiscretizer]:
    "Load Q-Learning or Dyna-Q agent."
    path = get_model_save_path(agent_name)
    filepath = os.path.join(path, f"{model_name}.pkl")

    with open(filepath, "rb") as f:
        save_data = pickle.load(f)

    # Recreate discretizer
    discretizer = StateDiscretizer(bins=save_data["discretizer_bins"])

    # Recreate agent with loaded Q-table
    agent = QLearningAgent(
        n_actions=save_data["n_actions"],
        epsilon=0.0,  # No exploration during evaluation
    )
    agent.Q.update(save_data["Q"])

    print(f"  Loaded model from: {filepath}")
    return agent, discretizer


def save_sb3_model(agent, agent_name: str, model_name: str) -> str:
    "Save Stable Baselines3 model (PPO, SAC)."
    path = get_model_save_path(agent_name)
    filepath = os.path.join(path, model_name)

    agent.save(filepath)
    print(f"  Saved model to: {filepath}.zip")
    return filepath


def load_ppo_model(model_name: str):
    "Load PPO model."
    from stable_baselines3 import PPO

    path = get_model_save_path("ppo")
    filepath = os.path.join(path, model_name)

    agent = PPO.load(filepath)
    print(f"  Loaded model from: {filepath}.zip")
    return agent


def load_sac_model(model_name: str):
    "Load SAC model."
    from stable_baselines3 import SAC

    path = get_model_save_path("sac")
    filepath = os.path.join(path, model_name)

    agent = SAC.load(filepath)
    print(f"  Loaded model from: {filepath}.zip")
    return agent


def save_ddqn_model(agent, agent_name: str, model_name: str) -> str:
    "Save Double DQN agent."
    import torch

    path = get_model_save_path(agent_name)
    filepath = os.path.join(path, f"{model_name}.pt")

    # Save the network state dict
    if hasattr(agent, "policy_net"):
        torch.save({
            "policy_net_state_dict": agent.policy_net.state_dict(),
            "target_net_state_dict": agent.target_net.state_dict(),
        }, filepath)
    elif hasattr(agent, "model"):
        torch.save({"model_state_dict": agent.model.state_dict()}, filepath)
    else:
        # Fallback: try to save the whole agent
        torch.save(agent, filepath)

    print(f"  Saved model to: {filepath}")
    return filepath


def load_ddqn_model(model_name: str):
    "Load Double DQN agent."

    path = get_model_save_path("DoubleDQN")
    filepath = os.path.join(path, f"{model_name}.pt")

    # Create a fresh agent and load weights
    env = ExtendedInventoryEnv_DDQN(discrete_actions=True)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = DoubleDQNAgent(state_dim=state_dim, action_dim=action_dim)

    checkpoint = torch.load(filepath, weights_only=False)
    if "policy_net_state_dict" in checkpoint:
        agent.policy_net.load_state_dict(checkpoint["policy_net_state_dict"])
        agent.target_net.load_state_dict(checkpoint["target_net_state_dict"])
    elif "model_state_dict" in checkpoint:
        agent.model.load_state_dict(checkpoint["model_state_dict"])

    print(f"  Loaded model from: {filepath}")
    return agent


def make_env_factory(
    env_cls: Callable[..., ExtendedInventoryEnv],
    **base_kwargs: Any,
) -> EnvFactory:
    """
    Create a factory that returns a fresh env instance each time.

    env_cls: the environment class (ExtendedInventoryEnv or ExtendedInventoryEnvPPO)
    base_kwargs: kwargs always passed to env constructor (e.g., discrete_actions=True)
    """
    def _factory(seed: Optional[int] = None) -> ExtendedInventoryEnv:
        kwargs = dict(base_kwargs)
        if seed is not None:
            kwargs["seed"] = seed
        return env_cls(**kwargs)

    return _factory


class InventoryEvaluator:
    """
    Evaluation framework for inventory management RL.

    Uses an env_factory to create a fresh environment instance for each
    evaluation episode. This avoids cross-episode contamination and
    makes evaluation fair across different agents.
    """

    def __init__(self, env_factory: EnvFactory):
        self.env_factory = env_factory

    def _get_initial_state(self, env) -> np.ndarray:
        out = env.reset()
        if isinstance(out, tuple):
            state, _info = out
        else:
            state = out
        return state

    def evaluate_episode(
        self,
        agent,
        discretizer: Optional[StateDiscretizer] = None,
        episode_seed: Optional[int] = None,
    ) -> Dict[str, float]:
        """
        Run a single evaluation episode and collect metrics.

        - For tabular / Dyna / Deep Q: pass a discretizer and Q-table / agent with select_action().
        - For PPO / SAC: discretizer=None, and agent must have .predict(obs, deterministic=True).
        """
        env = self.env_factory(episode_seed)
        state = self._get_initial_state(env)

        total_reward = 0.0
        total_cost = 0.0

        costs = {"holding": [], "stockout": [], "ordering": []}
        inventory_history: List[float] = []
        stockouts: List[int] = []
        lost_sales: List[float] = []
        demands: List[float] = []

        terminated = False
        truncated = False
        steps = 0
        max_steps = getattr(env, "episode_length", 365)

        while not (terminated or truncated) and steps < max_steps:

            if discretizer is not None:
                # Discrete (Q-Learning, Dyna-Q, Deep Q)
                disc_state = discretizer.discretize(state)

                if hasattr(agent, "Q"):
                    # Tabular Q-learning style: Q[(state, action)]
                    q_vals = [agent.Q[(disc_state, a)]
                              for a in range(agent.n_actions)]
                    action = int(np.argmax(q_vals))
                else:
                    # Generic discrete agent with select_action()
                    action = agent.select_action(disc_state)
            else:
                action, _ = agent.predict(state, deterministic=True)

            # Environment step
            next_state, reward, terminated, truncated, info = env.step(action)
            if isinstance(next_state, tuple):
                next_state, _info = next_state

            total_reward += reward
            total_cost += info["total_cost"]

            inv = info["inventory"]
            lost = info["lost_sales"]
            demand = info["demand"]
            order_q = info["order_quantity"]

            inventory_history.append(inv)
            stockouts.append(1 if lost > 0 else 0)
            lost_sales.append(lost)
            demands.append(demand)

            # Costs broken down
            holding_cost = env.holding_cost * inv
            stockout_cost = env.stockout_penalty * lost
            ordering_cost = (
                env.fixed_order_cost + env.variable_order_cost * order_q
                if order_q > 0
                else 0.0
            )

            costs["holding"].append(holding_cost)
            costs["stockout"].append(stockout_cost)
            costs["ordering"].append(ordering_cost)

            state = next_state
            steps += 1

        total_demand = float(sum(demands))
        total_lost = float(sum(lost_sales))
        fill_rate = 1.0 - \
            (total_lost / total_demand) if total_demand > 0 else 1.0

        return {
            "total_cost": total_cost,
            "total_reward": total_reward,
            "avg_cost": total_cost / steps if steps > 0 else 0.0,
            "holding_cost": float(sum(costs["holding"])),
            "stockout_cost": float(sum(costs["stockout"])),
            "ordering_cost": float(sum(costs["ordering"])),
            "avg_inventory": float(np.mean(inventory_history)) if inventory_history else 0.0,
            "stockout_rate": float(np.mean(stockouts)) if stockouts else 0.0,
            "fill_rate": fill_rate,
            "steps": steps,
        }

    def evaluate_multiple(
        self,
        agent,
        discretizer: Optional[StateDiscretizer] = None,
        num_episodes: int = 10,
        base_seed: int = 0,
    ) -> Dict[str, Dict[str, float]]:
        "Evaluate agent over multiple episodes with different seeds."

        results: List[Dict[str, float]] = []

        for i in range(num_episodes):
            metrics = self.evaluate_episode(
                agent, discretizer, episode_seed=base_seed + i)
            results.append(metrics)

        aggregated: Dict[str, Dict[str, float]] = {}
        if not results:
            return aggregated

        keys = results[0].keys()
        for key in keys:
            values = [r[key] for r in results]
            aggregated[key] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values))
            }

        return aggregated

    @staticmethod
    def print_report(metrics: Dict[str, Dict[str, float]], name: str = "Agent") -> None:
        "Print evaluation report."
        print(f"\n{name} Evaluation Report")

        print(f"\nCOSTS")
        print(
            f"  Avg Daily Cost: ${metrics['avg_cost']['mean']:.2f} (+/-{metrics['avg_cost']['std']:.2f})")
        print(f"  Holding Cost:   ${metrics['holding_cost']['mean']:.2f}")
        print(f"  Stockout Cost:  ${metrics['stockout_cost']['mean']:.2f}")
        print(f"  Ordering Cost:  ${metrics['ordering_cost']['mean']:.2f}")

        print(f"\nSERVICE LEVEL")
        print(f"  Fill Rate:     {metrics['fill_rate']['mean']:.1%}")
        print(f"  Stockout Rate: {metrics['stockout_rate']['mean']:.1%}")

        print(f"\nINVENTORY")
        print(f"  Avg Inventory: {metrics['avg_inventory']['mean']:.1f} units")

        print(f"\nPERFORMANCE")
        print(f"  Total Reward: {metrics['total_reward']['mean']:.2f}")

    @staticmethod
    def compare_algorithms(df: pd.DataFrame) -> None:
        "Plot comparison across algorithms based on a summary DataFrame."

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].bar(df["Algorithm"], df["Avg Cost"])
        axes[0].set_title("Average Daily Cost")
        axes[0].tick_params(axis="x", rotation=45)

        axes[1].bar(df["Algorithm"], df["Fill Rate"])
        axes[1].set_title("Fill Rate")
        axes[1].set_ylim([0, 1.05])
        axes[1].tick_params(axis="x", rotation=45)

        axes[2].bar(df["Algorithm"], df["Total Reward"])
        axes[2].set_title("Total Reward")
        axes[2].tick_params(axis="x", rotation=45)

        plt.tight_layout()
        plt.savefig("comparison.png", dpi=300)
        plt.show()


class DDQNWrapper:
    "Wrapper to make DDQN agent compatible with SB3-style predict() interface."

    def __init__(self, agent):
        self.agent = agent

    def predict(self, state, deterministic=True):
        return (self.agent.predict(state), None)


def main():
    """
    Main evaluation script:
      - For each algorithm, prompts to train new or load existing model
      - Saves trained models automatically
      - Evaluates each with fresh envs and shared metrics
    """

    results_rows: List[Dict[str, float]] = []

    # Q-LEARNING (TABULAR, DISCRETE ACTIONS)
    train_new, model_name = prompt_train_or_load("qlearning")

    if train_new:
        print("\n  Training Q-Learning agent...")
        q_agent, q_disc = train_q_agent(num_episodes=548)
        save_qlearning_model(q_agent, q_disc, "qlearning", model_name)
    else:
        q_agent, q_disc = load_qlearning_model("qlearning", model_name)

    q_env_factory = make_env_factory(
        ExtendedInventoryEnv, discrete_actions=True)
    q_evaluator = InventoryEvaluator(q_env_factory)
    q_metrics = q_evaluator.evaluate_multiple(q_agent, q_disc, num_episodes=10)
    q_evaluator.print_report(q_metrics, "Q-Learning")

    results_rows.append({
        "Algorithm": "Q-Learning",
        "Avg Cost": q_metrics["avg_cost"]["mean"],
        "Fill Rate": q_metrics["fill_rate"]["mean"],
        "Stockout Rate": q_metrics["stockout_rate"]["mean"],
        "Avg Inventory": q_metrics["avg_inventory"]["mean"],
        "Total Reward": q_metrics["total_reward"]["mean"],
    })

    # PPO (CONTINUOUS)
    train_new, model_name = prompt_train_or_load("ppo")

    if train_new:
        print("\n  Training PPO agent...")
        ppo_agent, _ = train_ppo_agent(
            num_timesteps=365_000,
            n_steps=512,
            learning_rate=2e-4,
            batch_size=64,
            n_epochs=15,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.001,
            seed=42,
            save_name=None,
        )
        save_sb3_model(ppo_agent, "ppo", model_name)
    else:
        ppo_agent = load_ppo_model(model_name)

    ppo_env_factory = make_env_factory(ExtendedInventoryEnvPPO)
    ppo_evaluator = InventoryEvaluator(ppo_env_factory)
    ppo_metrics = ppo_evaluator.evaluate_multiple(
        ppo_agent, discretizer=None, num_episodes=10)
    ppo_evaluator.print_report(ppo_metrics, "PPO")

    results_rows.append({
        "Algorithm": "PPO",
        "Avg Cost": ppo_metrics["avg_cost"]["mean"],
        "Fill Rate": ppo_metrics["fill_rate"]["mean"],
        "Stockout Rate": ppo_metrics["stockout_rate"]["mean"],
        "Avg Inventory": ppo_metrics["avg_inventory"]["mean"],
        "Total Reward": ppo_metrics["total_reward"]["mean"],
    })

    # SAC (CONTINUOUS)
    train_new, model_name = prompt_train_or_load("sac")

    if train_new:
        print("\n  Training SAC agent...")
        sac_agent, _ = train_sac_agent(num_timesteps=365_000)
        save_sb3_model(sac_agent, "sac", model_name)
    else:
        sac_agent = load_sac_model(model_name)

    sac_env_factory = make_env_factory(ExtendedInventoryEnvSAC)
    sac_evaluator = InventoryEvaluator(sac_env_factory)
    sac_metrics = sac_evaluator.evaluate_multiple(
        sac_agent, discretizer=None, num_episodes=10)
    sac_evaluator.print_report(sac_metrics, "SAC")

    results_rows.append({
        "Algorithm": "SAC",
        "Avg Cost": sac_metrics["avg_cost"]["mean"],
        "Fill Rate": sac_metrics["fill_rate"]["mean"],
        "Stockout Rate": sac_metrics["stockout_rate"]["mean"],
        "Avg Inventory": sac_metrics["avg_inventory"]["mean"],
        "Total Reward": sac_metrics["total_reward"]["mean"],
    })

    # DYNA-Q (DISCRETE)
    train_new, model_name = prompt_train_or_load("dyna_q")

    if train_new:
        print("\n  Training Dyna-Q agent...")
        dyna_agent, dyna_disc = train_dyna_agent(num_episodes=1000)
        save_qlearning_model(dyna_agent, dyna_disc, "dyna_q", model_name)
    else:
        dyna_agent, dyna_disc = load_qlearning_model("dyna_q", model_name)

    dyna_env_factory = make_env_factory(
        ExtendedInventoryEnv, discrete_actions=True)
    dyna_evaluator = InventoryEvaluator(dyna_env_factory)
    dyna_metrics = dyna_evaluator.evaluate_multiple(
        dyna_agent, dyna_disc, num_episodes=10)
    dyna_evaluator.print_report(dyna_metrics, "Dyna-Q")

    results_rows.append({
        "Algorithm": "Dyna-Q",
        "Avg Cost": dyna_metrics["avg_cost"]["mean"],
        "Fill Rate": dyna_metrics["fill_rate"]["mean"],
        "Stockout Rate": dyna_metrics["stockout_rate"]["mean"],
        "Avg Inventory": dyna_metrics["avg_inventory"]["mean"],
        "Total Reward": dyna_metrics["total_reward"]["mean"],
    })

    # DOUBLE DQN (DISCRETE)
    train_new, model_name = prompt_train_or_load("DoubleDQN")

    if train_new:
        print("\n  Training Double DQN agent...")
        ddqn_agent, _ = train_ddqn(num_episodes=1000)
        save_ddqn_model(ddqn_agent, "DoubleDQN", model_name)
    else:
        ddqn_agent = load_ddqn_model(model_name)

    wrapped_ddqn = DDQNWrapper(ddqn_agent)
    ddqn_env_factory = make_env_factory(
        ExtendedInventoryEnv_DDQN, discrete_actions=True)
    ddqn_evaluator = InventoryEvaluator(ddqn_env_factory)
    ddqn_metrics = ddqn_evaluator.evaluate_multiple(
        wrapped_ddqn, discretizer=None, num_episodes=10)
    ddqn_evaluator.print_report(ddqn_metrics, "Double DQN")

    results_rows.append({
        "Algorithm": "Double DQN",
        "Avg Cost": ddqn_metrics["avg_cost"]["mean"],
        "Fill Rate": ddqn_metrics["fill_rate"]["mean"],
        "Stockout Rate": ddqn_metrics["stockout_rate"]["mean"],
        "Avg Inventory": ddqn_metrics["avg_inventory"]["mean"],
        "Total Reward": ddqn_metrics["total_reward"]["mean"],
    })

    # Summary comparison
    df = pd.DataFrame(results_rows).sort_values("Avg Cost", ascending=True)
    print(" SUMMARY COMPARISON ")
    print(df.to_string(index=False))

    InventoryEvaluator.compare_algorithms(df)


if __name__ == "__main__":
    main()
