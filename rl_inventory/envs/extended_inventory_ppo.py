# rl_inventory/envs/extended_inventory_ppo.py

import numpy as np
from gymnasium.spaces import Box

from rl_inventory.envs.extended_inventory import ExtendedInventoryEnv


class ExtendedInventoryEnvPPO(ExtendedInventoryEnv):
    """
    PPO-compatible version of ExtendedInventoryEnv with continuous action support.

    PPO will output actions in [-1, 1]. This wrapper rescales them to [0, max_order]
    (here 0–300) before passing them to the base environment.
    """

    def __init__(self, *args, **kwargs):
        # Always force continuous-actions for this wrapper
        kwargs["discrete_actions"] = False
        super().__init__(*args, **kwargs)

        # PPO works best with normalized actions, so expose [-1, 1]
        self.action_space = Box(
            low=-1.0,
            high=1.0,
            shape=(1,),
            dtype=np.float32,
        )

        # Keep a single definition of the max order; must match the base env
        self._max_order = 300.0

    def _rescale_action(self, action) -> float:
        """
        Convert a PPO action in [-1, 1] to an actual order quantity in [0, max_order].
        """
        # Make sure we can handle floats, lists, or small arrays
        a = float(np.asarray(action).reshape(-1)[0])
        # Map [-1, 1] -> [0, max_order]
        # a = -1   -> 0
        # a =  1   -> max_order
        scaled = (a + 1.0) * 0.5 * self._max_order
        return float(np.clip(scaled, 0.0, self._max_order))

    def step(self, action):
        """
        Take one RL step.

        - `action` comes from PPO as a value in [-1, 1].
        - We rescale it to [0, max_order].
        - We pass it to the base environment as a 1D array, which it expects
          in continuous mode (`discrete_actions=False`).
        """
        order_quantity = self._rescale_action(action)
        # Base env expects an array-like in continuous mode and will read [0]
        return super().step(np.array([order_quantity], dtype=np.float32))
