"""
Evaluation Framework for Inventory Management RL

- Supports multiple algorithms (Q-Learning, PPO, SAC, etc.)
- Uses fresh environments per evaluation episode via env_factory
- Ensures fair, comparable evaluation across algorithms
"""

from typing import Callable, Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from rl_inventory.envs.extended_inventory import ExtendedInventoryEnv
from rl_inventory.envs.extended_inventory_ppo import ExtendedInventoryEnvPPO

from rl_inventory.agents.qlearning.qlearning import QLearningAgent, StateDiscretizer
from rl_inventory.scripts.q_learning_demo import train_agent as train_q_agent
from rl_inventory.scripts.ppo_demo import train_agent as train_ppo_agent
from rl_inventory.scripts.sac_demo import train_agent as train_sac_agent
from rl_inventory.scripts.dyna_q_demo import train_agent as train_dyna_agent



EnvFactory = Callable[[Optional[int]], ExtendedInventoryEnv]


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
            # all your envs accept seed in the constructor
            kwargs["seed"] = seed
        return env_cls(**kwargs)

    return _factory



# InventoryEvaluator


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
                    q_vals = [agent.Q[(disc_state, a)] for a in range(agent.n_actions)]
                    action = int(np.argmax(q_vals))
                else:
                    # Generic discrete agent with select_action()
                    action = agent.select_action(disc_state)
            else:
                
                action, _ = agent.predict(state, deterministic=True)

            #Environment step
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
        fill_rate = 1.0 - (total_lost / total_demand) if total_demand > 0 else 1.0

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
            metrics = self.evaluate_episode(agent, discretizer, episode_seed=base_seed + i)
            results.append(metrics)

        aggregated: Dict[str, Dict[str, float]] = {}
        if not results:
            return aggregated

        keys = results[0].keys()
        for key in keys:
            values = [r[key] for r in results]
            aggregated[key] = {"mean": float(np.mean(values)), "std": float(np.std(values))}

        return aggregated

    @staticmethod
    def print_report(metrics: Dict[str, Dict[str, float]], name: str = "Agent") -> None:
        "Print evaluation report."
        print(f"{name} Evaluation Report")

        print(f"\nCOSTS")
        print(f"  Avg Daily Cost: ${metrics['avg_cost']['mean']:.2f} (±{metrics['avg_cost']['std']:.2f})")
        print(f"  Holding Cost:   ${metrics['holding_cost']['mean']:.2f}")
        print(f"  Stockout Cost:  ${metrics['stockout_cost']['mean']:.2f}")
        print(f"  Ordering Cost:  ${metrics['ordering_cost']['mean']:.2f}")

        print(f"\nSERVICE LEVEL")
        print(f"  Fill Rate:   {metrics['fill_rate']['mean']:.1%}")
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


def main():
    """
    Example main:
      - Train Q-Learning (discrete)
      - Train PPO (continuous wrapper)
      - Train SAC (continuous plain)
      - Evaluate each with fresh envs and shared metrics

    Adjust num_episodes / num_timesteps to match your chosen fairness budget.
    """

    results_rows: List[Dict[str, float]] = []

    
    # 1. Q-LEARNING (TABULAR, DISCRETE ACTIONS)

    print(" Q-Learning (discrete) ")

    q_agent, q_disc = train_q_agent(num_episodes=548)

    q_env_factory = make_env_factory(ExtendedInventoryEnv, discrete_actions=True)
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

    
    print("\n PPO (Continous)")
   
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

    ppo_env_factory = make_env_factory(ExtendedInventoryEnvPPO)
    ppo_evaluator = InventoryEvaluator(ppo_env_factory)

    ppo_metrics = ppo_evaluator.evaluate_multiple(ppo_agent, discretizer=None, num_episodes=10)
    ppo_evaluator.print_report(ppo_metrics, "PPO")

    results_rows.append({
        "Algorithm": "PPO",
        "Avg Cost": ppo_metrics["avg_cost"]["mean"],
        "Fill Rate": ppo_metrics["fill_rate"]["mean"],
        "Stockout Rate": ppo_metrics["stockout_rate"]["mean"],
        "Avg Inventory": ppo_metrics["avg_inventory"]["mean"],
        "Total Reward": ppo_metrics["total_cost"]["mean"],
    })


    print("\n SAC (Continous)")
    sac_agent, _ = train_sac_agent(num_timesteps=365_000)

    sac_env_factory = make_env_factory(ExtendedInventoryEnv, discrete_actions=False)
    sac_evaluator = InventoryEvaluator(sac_env_factory)

    sac_metrics = sac_evaluator.evaluate_multiple(sac_agent, discretizer=None, num_episodes=10)
    sac_evaluator.print_report(sac_metrics, "SAC")

    results_rows.append({
        "Algorithm": "SAC",
        "Avg Cost": sac_metrics["avg_cost"]["mean"],
        "Fill Rate": sac_metrics["fill_rate"]["mean"],
        "Stockout Rate": sac_metrics["stockout_rate"]["mean"],
        "Avg Inventory": sac_metrics["avg_inventory"]["mean"],
        "Total Reward": sac_metrics["total_reward"]["mean"],
    })

   
    print("\n Dyna-Q (discrete)")
    
    dyna_agent, dyna_disc = train_dyna_agent(num_episodes=1000)
    dyna_env_factory = make_env_factory(ExtendedInventoryEnv, discrete_actions=True)
    dyna_evaluator = InventoryEvaluator(dyna_env_factory)
    dyna_metrics = dyna_evaluator.evaluate_multiple(dyna_agent, dyna_disc, num_episodes=10)
    dyna_evaluator.print_report(dyna_metrics, "Dyna-Q")

    results_rows.append({
        "Algorithm": "DYNA Q",
        "Avg Cost": dyna_metrics["avg_cost"]["mean"],
        "Fill Rate": dyna_metrics["fill_rate"]["mean"],
        "Stockout Rate": dyna_metrics["stockout_rate"]["mean"],
        "Avg Inventory": dyna_metrics["avg_inventory"]["mean"],
        "Total Reward": dyna_metrics["total_reward"]["mean"],
    })

    df = pd.DataFrame(results_rows).sort_values("Avg Cost", ascending=True)
    print("\n Summary Comparison")
    print(df.to_string(index=False))

    InventoryEvaluator.compare_algorithms(df)


if __name__ == "__main__":
    main()
