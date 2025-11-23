"""
SAC Agent using Stable Baselines 3 for Inventory Management
"""

import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
import torch


class TrainingCallback(BaseCallback):
    """Custom callback for tracking training metrics."""
    
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_costs = []
        
    def _on_step(self) -> bool:
        """Called at each environment step."""
        if self.locals.get('dones')[0]:
            info = self.locals['infos'][0]
            if 'episode' in info:
                episode_reward = info['episode']['r']
                self.episode_rewards.append(episode_reward)
                
                if len(self.episode_rewards) % 1000 == 0:
                    recent_rewards = self.episode_rewards[-10:]
                    avg_reward = np.mean(recent_rewards)
                    print(f"Episode {len(self.episode_rewards):4d} | "
                          f"Avg Reward (last 10): {avg_reward:8.2f}")
        
        return True
    
    
class SACAgent:
    def __init__(self, model):
        self.model = model
        
    def predict(self, state, deterministic=True):
        return self.model.predict(state, deterministic=deterministic)
    
    def select_action(self, state, deterministic=True):
        action, _ = self.predict(state, deterministic)
        return action
    
def train_sac(
        env_fn,
        learning_rate=3e-4,
        buffer_size=100000,
        learning_starts=1000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        seed=42,
        total_timesteps=200000,
        save_path="sac_inventory",
        verbose=1):
    print("Creating SAC model...")
    print(f"Total timesteps: {total_timesteps}")
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    env = env_fn()
    
    model = SAC(
        policy="MlpPolicy",
        env=env,
        learning_rate=learning_rate,
        buffer_size=buffer_size,
        learning_starts=learning_starts,
        batch_size=batch_size,
        tau=tau,
        gamma=gamma,
        verbose=verbose,
        seed=seed,
        device="auto",
    )
    
    callback = TrainingCallback(verbose=0)
    
    print("\nTraining SAC...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=callback,
        progress_bar=True
    )
    
    model.save(save_path)
    print(f"\nModel saved to {save_path}.zip")
    
    return model, callback


def load_sac_model(path="sac_inventory"):
    model = SAC.load(path)
    print(f"Model loaded from {path}.zip")
    return model