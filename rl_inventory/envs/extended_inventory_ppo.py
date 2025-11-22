# rl_inventory/envs/extended_inventory_ppo.py

import numpy as np
from gymnasium.spaces import Box

from rl_inventory.envs.extended_inventory import ExtendedInventoryEnv

class ExtendedInventoryEnvPPO(ExtendedInventoryEnv):
    """
    PPO-compatible version of ExtendedInventoryEnv with continuous action support.

    PPO will output actions in [0, 1]. This wrapper rescales them to [0, max_order]
    before passing them to the base environment.
    
    Key improvements:
    - Action space is [0, 1] instead of [-1, 1] to avoid negative action bias
    - Reward scaling to stabilize learning
    - Better action rescaling
    """

    def __init__(self, *args, **kwargs):
        self.enable_reward_shaping = kwargs.pop("reward_shaping", False)
        self.reward_scale = kwargs.pop("reward_scale", 0.01)  # Scale rewards
        
        # Always force continuous-actions for this wrapper
        kwargs["discrete_actions"] = False
        super().__init__(*args, **kwargs)

        # Use [0, 1] action space - easier for PPO to learn
        self.action_space = Box(
            low=0.0,
            high=1.0,
            shape=(1,),
            dtype=np.float32,
        )

        # Keep a single definition of the max order
        self._max_order = 300.0

    def reset(self, *, seed=None, options=None):
        """
        Reset the environment.
        
        This wrapper method handles the seed parameter that SB3 expects,
        even if the base environment doesn't support it.
        
        Args:
            seed: Random seed (handled here to maintain compatibility)
            options: Additional reset options
            
        Returns:
            Tuple of (observation, info) for Gymnasium compatibility
        """
        # Handle seed if provided
        if seed is not None:
            np.random.seed(seed)
        
        # Call the base reset without the seed parameter
        result = super().reset()
        
        # Ensure we return (observation, info) tuple
        if isinstance(result, tuple):
            return result
        else:
            return result, {}

    def _rescale_action(self, action) -> float:
        """
        Convert a PPO action in [0, 1] to an actual order quantity in [0, max_order].
        
        This is simpler than [-1, 1] rescaling and avoids the agent getting stuck
        at negative values that map to zero orders.
        """
        # Handle different input types
        a = float(np.asarray(action).reshape(-1)[0])
        # Clip to valid range and scale
        a = np.clip(a, 0.0, 1.0)
        scaled = a * self._max_order
        return float(scaled)

    def step(self, action):
        """
        Take one RL step with reward scaling.

        - `action` comes from PPO as a value in [0, 1].
        - We rescale it to [0, max_order].
        - We pass it to the base environment as a 1D array.
        - We scale the reward for more stable learning.
        """
        order_quantity = self._rescale_action(action)
        
        # Create a continuous action array for the base environment
        continuous_action = np.array([order_quantity], dtype=np.float32)

        # Get the base step result
        obs, reward, terminated, truncated, info = super().step(continuous_action)
        
        # Scale reward for better learning stability
        scaled_reward = reward * self.reward_scale
        
        return obs, scaled_reward, terminated, truncated, info