# rl_inventory/envs/extended_inventory_sac.py

import numpy as np
from gymnasium.spaces import Box

from rl_inventory.envs.extended_inventory import ExtendedInventoryEnv


class ExtendedInventoryEnvSAC(ExtendedInventoryEnv):
    "SAC-compatible version of ExtendedInventoryEnv."

    def __init__(self, *args, **kwargs):
        self.reward_scale = kwargs.pop("reward_scale", 0.01)
        
        # Force continuous actions for SAC
        kwargs["discrete_actions"] = False
        super().__init__(*args, **kwargs)

        # SAC works best with [0, 1] action space
        self.action_space = Box(
            low=0.0,
            high=1.0,
            shape=(1,),
            dtype=np.float32,
        )

        # Max order quantity
        self._max_order = 300.0

    def reset(self, *, seed=None, options=None):
        """
        Reset the environment.
        
        Gymnasium-compatible reset that accepts seed parameter.
        
        Args:
            seed: Random seed for reproducibility
            
        Returns:
            Tuple of (observation, info) for Gymnasium compatibility
        """
        # Handle seed if provided
        if seed is not None:
            np.random.seed(seed)
            # Also set environment's random state if it has one
            if hasattr(self, 'np_random'):
                self.np_random = np.random.RandomState(seed)
        
        result = super().reset()
        
        if isinstance(result, tuple):
            return result
        else:
            # If base only returns observation, add empty info dict
            return result, {}

    def _rescale_action(self, action) -> float:
        "Convert SAC action in [0, 1] to actual order quantity in [0, max_order]."
        
        # Handle different input types (float, list, array)
        a = float(np.asarray(action).reshape(-1)[0])
        
        # Clip to valid range
        a = np.clip(a, 0.0, 1.0)
        
        # Scale to order quantity
        scaled = a * self._max_order
        
        return float(scaled)

    def step(self, action):
        "Take one step in the environment."
        
        # Rescale action from [0, 1] to [0, max_order]
        order_quantity = self._rescale_action(action)
        
        # Create continuous action array for base environment
        continuous_action = np.array([order_quantity], dtype=np.float32)

        # Call base step
        obs, reward, terminated, truncated, info = super().step(continuous_action)
        
        # Scale reward if needed
        scaled_reward = reward * self.reward_scale
        
        return obs, scaled_reward, terminated, truncated, info