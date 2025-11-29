"PPO Agent using Stable Baselines 3 for Inventory Management"

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
import torch


class PPOAgent:
    "Wrapper for Stable Baselines 3 PPO."
    
    def __init__(self, model):
        self.model = model
        
    def predict(self, state, deterministic=True):
        "Predict action for given state."
        return self.model.predict(state, deterministic=deterministic)


def train_ppo(
        env_fn,
        total_timesteps,
        learning_rate,
        n_steps,
        batch_size,
        n_epochs,
        gamma,
        gae_lambda,
        clip_range,
        ent_coef,
        seed,
        save_path,
        verbose=0):
    
    # Set seeds
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Wrap environment in Monitor and DummyVecEnv (required by SB3)
    def make_env():
        return Monitor(env_fn())

    vec_env = DummyVecEnv([make_env])

    print("Creating PPO model...")
    print(f"Total timesteps: {total_timesteps}")
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")

    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        ent_coef=ent_coef,
        vf_coef=0.5,
        max_grad_norm=0.5,
        normalize_advantage=True,
        seed=seed,
        device="auto",
        verbose=verbose,
    )

    # Train
    print("\nTraining PPO...")
    model.learn(total_timesteps=total_timesteps, progress_bar=True)
    
    # Save
    model.save(save_path)
    print(f"\nModel saved to {save_path}.zip")
    
    return model, None


def load_ppo_model(path="ppo_inventory"):
    "Load a saved PPO model."
    return PPO.load(path)