import numpy as np
from typing import Tuple
from collections import defaultdict

class StateDiscretizer:
    
    def __init__(self, 
                 inventory_bins: int = 10,
                 pipeline_bins: int = 5, 
                 demand_bins: int = 5):

        self.inventory_bins = inventory_bins
        self.pipeline_bins = pipeline_bins
        self.demand_bins = demand_bins
        
        self.inventory_edges = np.linspace(0, 1, inventory_bins + 1)
        self.pipeline_edges = np.linspace(0, 1, pipeline_bins + 1)
        self.demand_edges = np.linspace(0, 1, demand_bins + 1)
    
    def discretize(self, state: np.ndarray) -> Tuple[int, int, int, int, int]:

        inventory = state[0]            # normalized inventory level
        pipeline = state[1]             # normalized pipeline stock
        recent_demand = np.mean(state[2:9])  # average of past 7 days of demand
        weather = int(state[9] * 2)     # convert 0.0/0.5/1.0 to 0/1/2
        promotion = int(state[10])      # 0 = no promo, 1 = promo active
        
        # Discretize into bins
        inv_bin = np.digitize(inventory, self.inventory_edges) - 1
        inv_bin = np.clip(inv_bin, 0, self.inventory_bins - 1)
        
        pipe_bin = np.digitize(pipeline, self.pipeline_edges) - 1
        pipe_bin = np.clip(pipe_bin, 0, self.pipeline_bins - 1)
        
        demand_bin = np.digitize(recent_demand, self.demand_edges) - 1
        demand_bin = np.clip(demand_bin, 0, self.demand_bins - 1)
        
        return (inv_bin, pipe_bin, demand_bin, weather, promotion)


class QLearningAgent:
    def __init__(self,
                 n_actions: int,
                 alpha: float = 0.1,     
                 gamma: float = 0.99,    
                 epsilon: float = 0.2):  
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

        self.Q = defaultdict(float)

    def select_action(self, state_key):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)
        else:
            q_vals = []
            for a in range(self.n_actions):
                q_vals.append(self.Q[(state_key, a)])
            return int(np.argmax(q_vals))

    def update(self, state, action, reward, new_state, terminated):
        old_q = self.Q[(state, action)]
        if terminated:
            target = reward
        else:
            next_qs = []
            for action in range(self.n_actions):
                next_qs.append(self.Q[(new_state, action)])
            target = reward + self.gamma * np.max(next_qs)

        self.Q[(state, action)] = old_q + self.alpha * (target - old_q)