from rl_inventory.envs.extended_inventory_sac import ExtendedInventoryEnvSAC
from stable_baselines3 import SAC


def train_agent(num_timesteps = 200000, learning_rate = 3e-4, buffer_size = 100000, verbose = 1):
    env = ExtendedInventoryEnvSAC(discrete_actions=False)
 
    agent = SAC(
        policy="MlpPolicy",          
        env=env,
        learning_rate=learning_rate,
        buffer_size=buffer_size,
        learning_starts=1000,         
        batch_size=256,               
        tau=0.005,                    
        gamma=0.99,                  
        verbose=verbose,
        device="cuda"
    )
    
    print(f"Training SAC for {num_timesteps} timesteps")
    agent.learn(total_timesteps=num_timesteps, progress_bar=True)
    
    agent.save("sac")
    
    return agent


def demo_run(agent):
    env = ExtendedInventoryEnvSAC(discrete_actions=False)
    
    state, info = env.reset()
    total_reward = 0.0
    

    for day in range(env.episode_length):
        action, s = agent.predict(state, deterministic=True)
        
        next_state, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        state = next_state
        
        if terminated or truncated:
            break
    
  
    print(f"\n SAC Training finished. Total reward = {total_reward:.2f}")
    env.plot_episode()


def main():
    agent = train_agent(num_timesteps=365_000, verbose=0)
    demo_run(agent)


if __name__ == "__main__":
    main()