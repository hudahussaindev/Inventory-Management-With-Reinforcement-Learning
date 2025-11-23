"""
PPO Agent using Stable Baselines 3 for Inventory Management
"""

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
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
                
                if len(self.episode_rewards) % 10 == 0:
                    recent_rewards = self.episode_rewards[-10:]
                    avg_reward = np.mean(recent_rewards)
                    print(f"Episode {len(self.episode_rewards):4d} | "
                          f"Avg Reward (last 10): {avg_reward:8.2f}")
        
        return True


class PPOAgent:
    """
    Wrapper for Stable Baselines 3 PPO to match the interface expected by evaluator.
    """
    
    def __init__(self, model):
        """
        Args:
            model: Trained SB3 PPO model
        """
        self.model = model
        
    def predict(self, state, deterministic=True):
        """
        Predict action for given state.
        
        Args:
            state: Current state observation
            deterministic: Whether to use deterministic policy
            
        Returns:
            Tuple of (action, None) to match SB3 interface
        """
        return self.model.predict(state, deterministic=deterministic)
    
    def select_action(self, state, deterministic=True):
        """Alternative method name for compatibility."""
        action, _ = self.predict(state, deterministic)
        return action


def create_ppo_model(env, 
                     learning_rate=3e-4,
                     n_steps=2048,
                     batch_size=64,
                     n_epochs=10,
                     gamma=0.99,
                     gae_lambda=0.95,
                     clip_range=0.2,
                     seed=42,
                     verbose=1):
    """
    Create a PPO model with specified hyperparameters.
    
    Args:
        env: The inventory environment
        learning_rate: Learning rate for optimizer
        n_steps: Number of steps per rollout
        batch_size: Minibatch size
        n_epochs: Number of update epochs per rollout
        gamma: Discount factor
        gae_lambda: GAE lambda parameter
        clip_range: PPO clipping epsilon
        seed: Random seed
        verbose: Verbosity level
        
    Returns:
        PPO model
    """
    # Set seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    if hasattr(env, 'seed'):
        env.seed(seed)
    
    # Wrap environment
    env = Monitor(env)
    env = DummyVecEnv([lambda: env])
    
    # Create model
    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=verbose,
        seed=seed,
        device='auto',
    )
    
    return model


def train_ppo(env,
              total_timesteps=200000,
              learning_rate=3e-4,
              n_steps=2048,
              batch_size=64,
              n_epochs=10,
              gamma=0.99,
              gae_lambda=0.95,
              clip_range=0.2,
              seed=42,
              save_path="ppo_inventory",
              verbose=1):
    """
    Train a PPO agent.
    
    Args:
        env: Training environment
        total_timesteps: Total training steps
        learning_rate: Learning rate
        n_steps: Steps per rollout
        batch_size: Minibatch size
        n_epochs: Update epochs
        gamma: Discount factor
        gae_lambda: GAE lambda
        clip_range: PPO clipping
        seed: Random seed
        save_path: Where to save model
        verbose: Verbosity
        
    Returns:
        Tuple of (trained_model, callback)
    """
    print("Creating PPO model...")
    print(f"Total timesteps: {total_timesteps}")
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    
    model = create_ppo_model(
        env=env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        seed=seed,
        verbose=verbose
    )
    
    # Create callback
    callback = TrainingCallback(verbose=0)
    
    # Train
    print("\nTraining PPO...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=callback,
        progress_bar=True
    )
    
    # Save
    model.save(save_path)
    print(f"\nModel saved to {save_path}.zip")
    
    return model, callback


def load_ppo_model(path="ppo_inventory"):
    """
    Load a saved PPO model.
    
    Args:
        path: Path to saved model (without .zip)
        
    Returns:
        Loaded PPO model
    """
    model = PPO.load(path)
    print(f"Model loaded from {path}.zip")
    return model