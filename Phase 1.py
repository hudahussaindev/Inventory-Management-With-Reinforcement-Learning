"""
This is a custom OpenAI Gym environment for inventory management that extends
the basic OR-Gym inventory environment with realistic external demand factors.

Based on:
- OpenAI Gym framework (https://gym.openai.com/)
- OR-Gym suite for operations research problems (https://github.com/hubbs5/or-gym)
- Standard inventory control theory

The environment simulates a single-product inventory system with:
1. Stochastic demand influenced by external factors
2. Variable lead times for replenishment
3. Cost trade-offs between holding inventory and stockouts
4. Realistic business conditions (weather, promotions, seasonality)

This serves as a foundation for reinforcement learning agents to learn
optimal inventory ordering policies under uncertainty.

Phase 1 Implementation

"""

import numpy as np
import gym
from gym import spaces
from typing import Tuple, Dict, Any
import matplotlib.pyplot as plt
from collections import deque


class ExtendedInventoryEnv(gym.Env):
    
    """
    Extended inventory environment with weather, promotions, and seasonality.
    This class implements a Markov Decision Process (MDP) for inventory control
    where an agent must decide how much to order at each time step to minimize
    total costs while meeting stochastic demand.
    
    STATE SPACE:
    - Current inventory level (normalized)
    - Pipeline inventory - orders in transit (normalized)
    - 7-day demand history (normalized) - allows agent to detect trends
    - Weather condition (0=Normal, 1=Adverse, 2=Favorable)
    - Promotion indicator (0=No promotion, 1=Active promotion)
    - Seasonality (sine-encoded to capture annual patterns)
    
    ACTION SPACE:
    - Discrete mode: 9 predefined order quantities [0, 10, 20, 30, 50, 75, 100, 150, 200]
    - Continuous mode: Any order quantity in [0, 300]
    
    REWARD:
    Negative cost (to be minimized):
    - Holding cost: Cost per unit per time step for stored inventory
    - Stockout penalty: High cost for each unit of unmet demand
    - Fixed ordering cost: One-time cost per order (economies of scale)
    - Variable ordering cost: Cost per unit ordered
    """
    
    def __init__(
         self,
        max_inventory: int = 500,        # Maximum storage capacity
        max_pipeline: int = 600,          # Maximum orders in transit
        max_demand: int = 200,            # Maximum possible demand per period
        lead_time_range: Tuple[int, int] = (2, 5),  # Min/max days for order delivery
        base_demand: float = 50.0,        # Average daily demand (lambda for Poisson)
        holding_cost: float = 1.0,        # Cost per unit per day to hold inventory
        stockout_penalty: float = 10.0,   # Penalty per unit of lost sales
        fixed_order_cost: float = 50.0,   # Fixed cost per order placed
        variable_order_cost: float = 2.0, # Cost per unit ordered
        episode_length: int = 365,        # Number of days per episode (1 year)
        discrete_actions: bool = True,    # Use discrete vs continuous actions
        seed: int = None                  # Random seed 
    ):
        super().__init__()
        
        # Environment parameters
        self.max_inventory = max_inventory
        self.max_pipeline = max_pipeline
        self.max_demand = max_demand
        self.lead_time_range = lead_time_range
        self.base_demand = base_demand
        self.episode_length = episode_length
        
        # Cost parameters
        self.holding_cost = holding_cost
        self.stockout_penalty = stockout_penalty
        self.fixed_order_cost = fixed_order_cost
        self.variable_order_cost = variable_order_cost
        
        # External factors parameters
        self.weather_impact = {0: 0.0, 1: -0.2, 2: 0.2}  # normal, adverse, favorable
        self.promotion_impact = 0.3
        self.seasonality_impact = 0.15
        
        # State components
        self.inventory = 0
        self.pipeline = 0
        self.demand_history = deque(maxlen=7)
        self.weather = 0
        self.promotion = 0
        self.time_step = 0
        
        # Pipeline queue for lead time management
        self.pipeline_queue = []
        
        # Define action space
        self.discrete_actions = discrete_actions
        if discrete_actions:
            self.action_space = spaces.Discrete(9)
            self.action_values = [0, 10, 20, 30, 50, 75, 100, 150, 200]
        else:
            self.action_space = spaces.Box(low=0, high=300, shape=(1,), dtype=np.float32)
        
        # Define observation space
        self.observation_space = spaces.Box(
            low=np.array([0, 0] + [0]*7 + [0, 0, -1]),
            high=np.array([max_inventory, max_pipeline] + [max_demand]*7 + [2, 1, 1]),
            dtype=np.float32
        )
        
        # Set random seed
        self.seed(seed)
        
        # Tracking for visualization
        self.history = {
            'inventory': [],
            'demand': [],
            'orders': [],
            'costs': [],
            'weather': [],
            'promotions': []
        }
    
    def seed(self, seed=None):
        """Set random seed for reproducibility."""
        self.np_random = np.random.RandomState(seed)
        return [seed]
    
    def reset(self) -> np.ndarray:
        """Reset environment to initial state."""
        # Initialize inventory levels
        self.inventory = self.np_random.randint(50, 150)
        self.pipeline = 0
        self.pipeline_queue = []
        
        # Initialize demand history with reasonable values
        self.demand_history.clear()
        for _ in range(7):
            demand = self.np_random.poisson(self.base_demand)
            self.demand_history.append(min(demand, self.max_demand))
        
        # Initialize external factors
        self.weather = 0  
        self.promotion = 0 
        self.time_step = 0
        
        # Clear history
        for key in self.history:
            self.history[key] = []
        
        return self._get_state()
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """Execute one environment step."""
        # Convert action to order quantity
        if self.discrete_actions:
            order_quantity = self.action_values[action]
        else:
            order_quantity = float(action[0])
        
        # Sample external factors
        self._update_external_factors()
        
        # Generate demand
        demand = self._generate_demand()
        
        # Process pipeline deliveries
        delivered = self._process_pipeline()
        
        # Update inventory
        self.inventory = min(self.inventory + delivered, self.max_inventory)
        satisfied_demand = min(demand, self.inventory)
        lost_sales = demand - satisfied_demand
        self.inventory = max(0, self.inventory - demand)
        
        # Place new order if quantity > 0
        if order_quantity > 0:
            lead_time = self.np_random.randint(self.lead_time_range[0], self.lead_time_range[1] + 1)
            self.pipeline_queue.append({
                'quantity': order_quantity,
                'arrival_time': self.time_step + lead_time
            })
            self.pipeline += order_quantity
        
        # Calculate costs (negative reward)
        holding = self.holding_cost * self.inventory
        stockout = self.stockout_penalty * lost_sales
        fixed = self.fixed_order_cost if order_quantity > 0 else 0
        variable = self.variable_order_cost * order_quantity
        total_cost = holding + stockout + fixed + variable
        reward = -total_cost
        
        # Update demand history
        self.demand_history.append(min(demand, self.max_demand))
        
        # Record history
        self.history['inventory'].append(self.inventory)
        self.history['demand'].append(demand)
        self.history['orders'].append(order_quantity)
        self.history['costs'].append(total_cost)
        self.history['weather'].append(self.weather)
        self.history['promotions'].append(self.promotion)
        
        # Increment time
        self.time_step += 1
        done = self.time_step >= self.episode_length
        
        # Info dictionary
        info = {
            'inventory': self.inventory,
            'demand': demand,
            'lost_sales': lost_sales,
            'order_quantity': order_quantity,
            'total_cost': total_cost,
            'weather': self.weather,
            'promotion': self.promotion
        }
        
        return self._get_state(), reward, done, info
    
    def _get_state(self) -> np.ndarray:
        """Get current state observation."""
        # Calculate seasonality (sine-encoded)
        seasonality = np.sin(2 * np.pi * self.time_step / 365)
        
        # Construct state vector
        state = np.array([
            self.inventory / self.max_inventory,  # Normalize
            self.pipeline / self.max_pipeline,    # Normalize
            *[d / self.max_demand for d in self.demand_history],  # Normalized history
            self.weather / 2.0,  # Normalize to [0, 1]
            self.promotion,
            seasonality
        ], dtype=np.float32)
        
        return state
    
    def _update_external_factors(self):
        """Update weather and promotion status."""
        # Weather transitions (Markov chain)
        if self.weather == 0:  # Normal
            probs = [0.7, 0.15, 0.15]
        elif self.weather == 1:  # Adverse
            probs = [0.3, 0.5, 0.2]
        else:  # Favorable
            probs = [0.3, 0.2, 0.5]
        
        self.weather = self.np_random.choice([0, 1, 2], p=probs)
        
        # Promotion events (random with higher probability during certain periods)
        base_promo_prob = 0.1
        # Increase probability during "holiday" periods
        if (self.time_step % 365) in range(330, 365) or (self.time_step % 365) in range(0, 30):
            base_promo_prob = 0.3
        
        self.promotion = 1 if self.np_random.random() < base_promo_prob else 0
    
    def _generate_demand(self) -> float:
        """Generate demand based on current conditions."""
        # Base demand with modifiers
        lambda_t = self.base_demand
        
        # Weather impact
        lambda_t *= (1 + self.weather_impact[self.weather])
        
        # Promotion impact
        if self.promotion:
            lambda_t *= (1 + self.promotion_impact)
        
        # Seasonality impact via sine wave
        seasonality = abs(np.sin(2 * np.pi * self.time_step / 365))
        lambda_t *= (1 + self.seasonality_impact * seasonality)
        
        # Sample from Poisson distribution
        demand = self.np_random.poisson(lambda_t)
        
        return demand
    
    def _process_pipeline(self) -> float:
        """Process pipeline deliveries for current time step."""
        delivered = 0
        remaining_orders = []
        
        for order in self.pipeline_queue:
            if order['arrival_time'] <= self.time_step:
                delivered += order['quantity']
                self.pipeline -= order['quantity']
            else:
                remaining_orders.append(order)
        
        self.pipeline_queue = remaining_orders
        return delivered
    
    def render(self, mode='human'):
        """Render the environment (for visualization)."""
        if mode == 'human':
            print(f"Time: {self.time_step}, Inventory: {self.inventory:.0f}, "
                  f"Pipeline: {self.pipeline:.0f}, Weather: {self.weather}, "
                  f"Promotion: {self.promotion}")
    
    def plot_episode(self):
        """Plot episode history for analysis."""
        if not self.history['inventory']:
            print("No history to plot. Run an episode first.")
            return
        
        fig, axes = plt.subplots(3, 2, figsize=(12, 10))
        
        # Inventory levels
        axes[0, 0].plot(self.history['inventory'], label='Inventory')
        axes[0, 0].set_title('Inventory Levels')
        axes[0, 0].set_xlabel('Time Step')
        axes[0, 0].set_ylabel('Units')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Demand
        axes[0, 1].plot(self.history['demand'], label='Demand', alpha=0.7)
        axes[0, 1].set_title('Demand Pattern')
        axes[0, 1].set_xlabel('Time Step')
        axes[0, 1].set_ylabel('Units')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Orders
        axes[1, 0].plot(self.history['orders'], label='Orders', color='green')
        axes[1, 0].set_title('Order Quantities')
        axes[1, 0].set_xlabel('Time Step')
        axes[1, 0].set_ylabel('Units')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Costs
        axes[1, 1].plot(self.history['costs'], label='Total Cost', color='red')
        axes[1, 1].set_title('Costs Over Time')
        axes[1, 1].set_xlabel('Time Step')
        axes[1, 1].set_ylabel('Cost ($)')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        # Weather
        axes[2, 0].plot(self.history['weather'], label='Weather', color='blue')
        axes[2, 0].set_title('Weather Conditions (0=Normal, 1=Adverse, 2=Favorable)')
        axes[2, 0].set_xlabel('Time Step')
        axes[2, 0].set_ylabel('Weather State')
        axes[2, 0].set_ylim(-0.5, 2.5)
        axes[2, 0].legend()
        axes[2, 0].grid(True)
        
        # Promotions
        axes[2, 1].plot(self.history['promotions'], label='Promotions', color='orange')
        axes[2, 1].set_title('Promotion Events')
        axes[2, 1].set_xlabel('Time Step')
        axes[2, 1].set_ylabel('Promotion Active')
        axes[2, 1].set_ylim(-0.1, 1.1)
        axes[2, 1].legend()
        axes[2, 1].grid(True)
        
        plt.tight_layout()
        plt.show()


# Testing utilities
def test_environment():
    """Test the extended inventory environment."""
    env = ExtendedInventoryEnv(discrete_actions=True, seed=42)
    
    print("Environment created successfully!")
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    
    # Run a random episode
    state = env.reset()
    print(f"\nInitial state shape: {state.shape}")
    print(f"Initial state: {state}")
    
    total_reward = 0
    for t in range(100):
        action = env.action_space.sample()
        state, reward, done, info = env.step(action)
        total_reward += reward
        
        if t % 20 == 0:
            env.render()
        
        if done:
            break
    
    print(f"\nEpisode finished after {t+1} steps")
    print(f"Total reward: {total_reward:.2f}")
    env.plot_episode()
    
    return env


if __name__ == "__main__":
    env = test_environment()
