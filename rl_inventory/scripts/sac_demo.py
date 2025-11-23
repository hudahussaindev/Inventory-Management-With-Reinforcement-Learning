"""
SAC Demo for Inventory Management
"""

import os
from rl_inventory.envs.extended_inventory_sac import ExtendedInventoryEnvSAC
from rl_inventory.agents.sac.SAC import train_sac, load_sac_model, SACAgent


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
MODELS_DIR = os.path.join(PROJECT_ROOT, "rl_inventory", "agents", "sac", "trained_models")
MODEL_PATH = os.path.join(MODELS_DIR, "sac_inventory")


def train_agent(num_timesteps=200000, learning_rate=3e-4, buffer_size=100000, verbose=0):
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    def make_env():
        return ExtendedInventoryEnvSAC(discrete_actions=False, seed=42)
    
    print(f"Training SAC for {num_timesteps:,}")
    
    model, callback = train_sac(
        env=make_env,
        total_timesteps=num_timesteps,
        learning_rate=learning_rate,
        buffer_size=buffer_size,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        seed=42,
        save_path=MODEL_PATH,
        verbose=verbose,
    )
    
    return SACAgent(model), None

def load_agent():
    print(f"\n Loading model from: {MODEL_PATH}.zip")
    model = load_sac_model(MODEL_PATH)
    return SACAgent(model), None


def demo_run(agent):
    env = ExtendedInventoryEnvSAC(discrete_actions=False, seed=42)
    state, info = env.reset()
    total_reward = 0
    
    print("\nRunning demonstration episode...")
    print(f"{'Day':<6} {'Inventory':<10} {'Order':<10} {'Demand':<10} {'Lost Sales':<12} {'Cost':<10}")
    
    for day in range(env.episode_length):
        action, _ = agent.predict(state, deterministic=True)
        
        next_state, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Print daily info
        if day % 20 == 0 or day < 10:
            print(
                f"{day:<6} {info['inventory']:<10.1f} {info['order_quantity']:<10.1f} "
                f"{info['demand']:<10.1f} {info['lost_sales']:<12.1f} ${info['total_cost']:<9.2f}"
            )
        
        state = next_state
        
        if terminated or truncated:
            break
    
    print(f"\nEPISODE SUMMARY:")
    print(f"   Total days: {day+1}")
    print(f"   Total reward: {(total_reward*100):.2f}")
    print(f"   Total cost: ${-(total_reward*100):.2f}")
    print(f"   Average daily cost: ${-(total_reward*100)/(day+1):.2f}")
    
    print("\nPlotting Episode...")
    env.plot_episode()
 
def main():
    print("\n SAC INVENTORY MANAGEMENT DEMO")
    if os.path.exists(MODEL_PATH + ".zip"):
        print("\nFound existing model")
        agent, _ = load_agent()
    else:
        print("\nNo model found. Training new SAC agent")
        agent, _ = train_agent(num_timesteps=200000, verbose=0)
    
    demo_run(agent)
    
if __name__ == "__main__":
    main()