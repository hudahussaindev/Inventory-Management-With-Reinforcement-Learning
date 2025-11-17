"""
Evaluation Framework for Inventory Management RL
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List
from rl_inventory.envs.extended_inventory import ExtendedInventoryEnv
from rl_inventory.agents.qlearning.qlearning import QLearningAgent, StateDiscretizer


class InventoryEvaluator:
    """Simplified evaluation framework for inventory management RL."""
    
    def __init__(self, env: ExtendedInventoryEnv = None):
        self.env = env if env else ExtendedInventoryEnv(discrete_actions=True, seed=42)
        
    def evaluate_episode(self, agent, discretizer=None, seed=None) -> Dict:
        """Run single evaluation episode and collect metrics."""
        if seed:
            self.env.seed(seed)
            
        state = self.env.reset()
        total_reward = total_cost = 0
        costs = {'holding': [], 'stockout': [], 'ordering': []}
        inventory = []
        stockouts = []
        lost_sales = []
        demands = []
        
        done = False
        steps = 0
        
        while not done and steps < self.env.episode_length:
            # Get action
            if discretizer:
                discrete_state = discretizer.discretize(state)
                if hasattr(agent, 'Q'):
                    q_vals = [agent.Q[(discrete_state, a)] for a in range(agent.n_actions)]
                    action = int(np.argmax(q_vals))
                else:
                    action = agent.select_action(discrete_state)
            else:
                action = agent.predict(state, deterministic=True)[0]
            
            # Step
            next_state, reward, terminated, truncated, info = self.env.step(action)
            
            # Collect metrics
            total_reward += reward
            total_cost += info['total_cost']
            costs['holding'].append(self.env.holding_cost * info['inventory'])
            costs['stockout'].append(self.env.stockout_penalty * info['lost_sales'])
            costs['ordering'].append(
                (self.env.fixed_order_cost + self.env.variable_order_cost * info['order_quantity'])
                if info['order_quantity'] > 0 else 0
            )
            inventory.append(info['inventory'])
            stockouts.append(1 if info['lost_sales'] > 0 else 0)
            lost_sales.append(info['lost_sales'])
            demands.append(info['demand'])
            
            state = next_state
            done = terminated or truncated
            steps += 1
        
        # Calculate metrics
        total_demand = sum(demands)
        total_lost = sum(lost_sales)
        fill_rate = 1 - (total_lost / total_demand) if total_demand > 0 else 1.0
        
        return {
            'total_cost': total_cost,
            'total_reward': total_reward,
            'avg_cost': total_cost / steps if steps > 0 else 0,
            'holding_cost': sum(costs['holding']),
            'stockout_cost': sum(costs['stockout']),
            'ordering_cost': sum(costs['ordering']),
            'avg_inventory': np.mean(inventory),
            'stockout_rate': np.mean(stockouts),
            'fill_rate': fill_rate,
            'steps': steps
        }
    
    def evaluate_multiple(self, agent, discretizer=None, num_episodes=10) -> Dict:
        """Evaluate agent over multiple episodes."""
        results = []
        for i in range(num_episodes):
            metrics = self.evaluate_episode(agent, discretizer, seed=i)
            results.append(metrics)
        
        # Aggregate results
        aggregated = {}
        for key in results[0].keys():
            values = [r[key] for r in results]
            aggregated[key] = {'mean': np.mean(values), 'std': np.std(values)}
        
        return aggregated
    
    def print_report(self, metrics: Dict, name: str = "Agent"):
        """Print evaluation report."""
        print(f"{name} Evaluation Report")
        print(f"\nCOSTS")
        print(f"  Avg Daily Cost: ${metrics['avg_cost']['mean']:.2f} (±{metrics['avg_cost']['std']:.2f})")
        print(f"  Holding Cost: ${metrics['holding_cost']['mean']:.2f}")
        print(f"  Stockout Cost: ${metrics['stockout_cost']['mean']:.2f}")
        print(f"  Ordering Cost: ${metrics['ordering_cost']['mean']:.2f}")
        print(f"\nSERVICE LEVEL")
        print(f"  Fill Rate: {metrics['fill_rate']['mean']:.1%}")
        print(f"  Stockout Rate: {metrics['stockout_rate']['mean']:.1%}")
        print(f"\nINVENTORY")
        print(f"  Avg Inventory: {metrics['avg_inventory']['mean']:.1f} units")
        print(f"\nPERFORMANCE")
        print(f"  Total Reward: {metrics['total_reward']['mean']:.2f}")
    
    def compare_algorithms(self, algorithms: Dict, num_episodes=10) -> pd.DataFrame:
        """Compare multiple algorithms."""
        results = []
        
        for name, (agent, discretizer) in algorithms.items():
            print(f"Evaluating {name}...")
            metrics = self.evaluate_multiple(agent, discretizer, num_episodes)
            results.append({
                'Algorithm': name,
                'Avg Cost': metrics['avg_cost']['mean'],
                'Fill Rate': metrics['fill_rate']['mean'],
                'Stockout Rate': metrics['stockout_rate']['mean'],
                'Avg Inventory': metrics['avg_inventory']['mean'],
                'Total Reward': metrics['total_reward']['mean']
            })
        
        df = pd.DataFrame(results).sort_values('Total Reward', ascending=False)
        
        # Plot comparison
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].bar(df['Algorithm'], df['Avg Cost'])
        axes[0].set_title('Average Daily Cost')
        axes[0].tick_params(axis='x', rotation=45)
        
        axes[1].bar(df['Algorithm'], df['Fill Rate'])
        axes[1].set_title('Fill Rate')
        axes[1].set_ylim([0, 1.05])
        axes[1].tick_params(axis='x', rotation=45)
        
        axes[2].bar(df['Algorithm'], df['Total Reward'])
        axes[2].set_title('Total Reward')
        axes[2].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('comparison.png', dpi=300)
        plt.show()
        
        return df


def main():
    env = ExtendedInventoryEnv(discrete_actions=True, seed=42)
    evaluator = InventoryEvaluator(env)
    
    # Train agent
    print("Training Q-Learning agent...")
    from rl_inventory.scripts.q_learning_demo import train_agent
    agent, disc = train_agent(num_episodes=200)
    
    # Evaluate
    print("\nEvaluating...")
    metrics = evaluator.evaluate_multiple(agent, disc, num_episodes=10)
    evaluator.print_report(metrics, "Q-Learning")


if __name__ == "__main__":
    main()
