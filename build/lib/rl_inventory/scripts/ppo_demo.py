"""
PPO Demo for Inventory Management
"""

import os
import glob
from rl_inventory.envs.extended_inventory import ExtendedInventoryEnv
from rl_inventory.agents.ppo.PPO import train_ppo, load_ppo_model, PPOAgent
import numpy as np


# Get paths relative to this file
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
MODELS_DIR = os.path.join(PROJECT_ROOT, "rl_inventory", "agents", "ppo", "trained_models")
DEFAULT_MODEL_NAME = "ppo_inventory"
print("MODELS_DIR being used:", MODELS_DIR)
print("ZIPs found there:", glob.glob(os.path.join(MODELS_DIR, "*.zip")))



def list_available_models(models_dir):
    """
    List all available trained models in the trained_models directory.
    
    Args:
        models_dir: Path to trained_models directory
        
    Returns:
        List of model names (without .zip extension)
    """
    # Find all .zip files in trained_models directory
    if not os.path.exists(models_dir):
        return []
    
    zip_files = glob.glob(os.path.join(models_dir, "*.zip"))
    # Extract just the model names (without path and .zip)
    model_names = [os.path.splitext(os.path.basename(f))[0] for f in zip_files]
    return sorted(model_names)


def train_agent(
    num_timesteps=200000,
    n_steps=2048,
    learning_rate=3e-4,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    seed=42,
    save_name=None
):
    """
    Train a PPO agent on the inventory environment.
    
    Args:
        num_timesteps: Total training timesteps
        n_steps: Rollout length
        learning_rate: Learning rate
        batch_size: Minibatch size
        n_epochs: Update epochs per rollout
        gamma: Discount factor
        gae_lambda: GAE lambda parameter
        clip_range: PPO clipping epsilon
        seed: Random seed
        save_name: Name for saved model (without .zip)
        
    Returns:
        Tuple of (PPOAgent wrapper, None)
    """
    # Create trained_models directory if it doesn't exist
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    # Determine save name
    if save_name is None:
        # Generate a name with timestamp
        import time
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        save_name = f"ppo_model_{timestamp}"
    
    save_path = os.path.join(MODELS_DIR, save_name)
    
    # Create environment (continuous actions for PPO)
    env = ExtendedInventoryEnv(discrete_actions=False, seed=seed)
    
    print("\n" + "="*70)
    print("TRAINING PPO AGENT\n")
    print(f"Total timesteps: {num_timesteps:,}")
    print(f"Rollout length: {n_steps}")
    print(f"Save location: {save_path}.zip")
    
    # Train PPO
    model, callback = train_ppo(
        env=env,
        total_timesteps=num_timesteps,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        seed=seed,
        save_path=save_path,
        verbose=1
    )
    
    # Wrap model in PPOAgent for consistent interface
    agent = PPOAgent(model)
    
    print(f"\nModel saved to: {save_path}.zip")
    
    # Return None for discretizer (PPO doesn't need one)
    return agent, None


def load_agent(path):
    """
    Load a trained PPO agent.
    
    Args:
        path: Path to saved model (without .zip)
        
    Returns:
        Tuple of (PPOAgent wrapper, None)
    """
    # Check if file exists
    if not os.path.exists(path + ".zip"):
        raise FileNotFoundError(
            f"Model not found: {path}.zip\n"
            f"Train a model first."
        )
    
    print(f"\nLoading model from: {path}.zip")
    model = load_ppo_model(path)
    agent = PPOAgent(model)
    print("Model loaded successfully!")
    return agent, None


def demo_run(agent, discretizer=None):
    """
    Run a demonstration episode with the trained agent.
    
    Args:
        agent: Trained PPO agent
        discretizer: Not used for PPO (included for compatibility)
    """
    # Create environment (continuous actions)
    env = ExtendedInventoryEnv(discrete_actions=False, seed=42)
    state, info = env.reset()
    total_reward = 0
    
    print("\nRunning demonstration episode...")
    print(f"{'Day':<6} {'Inventory':<10} {'Order':<10} {'Demand':<10} {'Lost Sales':<12} {'Cost':<10}")
    
    for day in range(env.episode_length):
        # Get action from PPO agent
        action, _ = agent.predict(state, deterministic=True)
        
        # Take action in environment
        next_state, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Print daily info
        if day % 20 == 0 or day < 10:  # Print first 10 days and every 20th day
            print(
                f"{day:<6} {info['inventory']:<10.1f} {info['order_quantity']:<10.1f} "
                f"{info['demand']:<10.1f} {info['lost_sales']:<12.1f} ${info['total_cost']:<9.2f}"
            )
        
        state = next_state
        
        if terminated or truncated:
            break
    
    print(f"\nEPISODE SUMMARY:")
    print(f"   Total days: {day+1}")
    print(f"   Total reward: {total_reward:.2f}")
    print(f"   Total cost: ${-total_reward:.2f}")
    print(f"   Average daily cost: ${-total_reward/(day+1):.2f}")
    
    # Plot episode
    print("\nPlotting Episode...")
    try:
        env.plot_episode()
    except Exception as e:
        print(f"Could not generate plot: {e}")


def main():
    """
    Main training and demonstration.
    
    Lists available models, lets user choose, or train new.
    """
    print("PPO INVENTORY MANAGEMENT DEMO")
    
    # Create trained_models directory if it doesn't exist
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    # List available models
    available_models = list_available_models(MODELS_DIR)
    
    if available_models:
        print(f"\nFound {len(available_models)} trained model(s) in trained_models/:")
        print()
        
        for i, model_name in enumerate(available_models, 1):
            model_path = os.path.join(MODELS_DIR, model_name + ".zip")
            print(f"  [{i}] {model_name}")
        
        print(f"  [0] Train a new model")
        print()
        
        # Get user choice
        while True:
            choice = input("Select a model to use (number or 'na' for new): ").strip().lower()
            
            if choice == 'na' or choice == '0':
                print("\nTraining new model...")
                
                custom_name = input("\nEnter model name (or press Enter for auto-generated name): ").strip()
                if custom_name:
                    agent, _ = train_agent(num_timesteps=200000, save_name=custom_name)
                else:
                    agent, _ = train_agent(num_timesteps=200000)
                break
            
            try:
                choice_num = int(choice)
                if 1 <= choice_num <= len(available_models):
                    selected_model = available_models[choice_num - 1]
                    model_path = os.path.join(MODELS_DIR, selected_model)
                    print(f"\n→ Loading model: {selected_model}")
                    agent, _ = load_agent(model_path)
                    break
                else:
                    print(f"Invalid choice. Please enter a number between 0 and {len(available_models)}, or 'na'")
            except ValueError:
                print("Invalid input. Please enter a number or 'na'")
    else:
        print("\n No trained models found in trained_models/")
        print("Training new model...")
        
        # Ask for custom name
        custom_name = input("\nEnter model name (or press Enter for 'ppo_inventory'): ").strip()
        if not custom_name:
            custom_name = "ppo_inventory"
        
        agent, _ = train_agent(num_timesteps=200000, save_name=custom_name)
    
    # Run demonstration
    demo_run(agent)

if __name__ == "__main__":
    main()