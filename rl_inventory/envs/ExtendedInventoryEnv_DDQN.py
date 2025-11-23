import numpy as np
from typing import Tuple, Dict
from .extended_inventory import ExtendedInventoryEnv


class ExtendedInventoryEnv_DDQN(ExtendedInventoryEnv):
    """ Wrapper around ExtendedInventoryEnv with reward scaling for Double DQN. """

    def __init__(
        self,
        reward_scale: float = 100.0,
        inventory_target_min: float = 40.0,
        inventory_target_max: float = 100.0,
        use_reward_shaping: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.reward_scale = reward_scale
        self.inventory_target_min = inventory_target_min
        self.inventory_target_max = inventory_target_max
        self.use_reward_shaping = use_reward_shaping

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        # Execute one environment step
        state, reward, terminated, truncated, info = super().step(action)

        inventory = info.get('inventory', 0)
        lost_sales = info.get('lost_sales', 0)

        scaled_reward = reward / self.reward_scale

        if self.use_reward_shaping:
            extra_stockout = lost_sales * 2.0 / self.reward_scale
            if self.inventory_target_min <= inventory <= self.inventory_target_max:
                inv_bonus = 0.05 / self.reward_scale
            elif inventory < self.inventory_target_min:
                inv_bonus = -0.02 / self.reward_scale
            else:
                inv_bonus = -0.03 / self.reward_scale
            scaled_reward = scaled_reward - extra_stockout + inv_bonus

        info['original_reward'] = reward
        info['scaled_reward'] = scaled_reward
        info['original_total_cost'] = info.get('total_cost', -reward)

        return state, scaled_reward, terminated, truncated, info

    def reset(self) -> np.ndarray:
        return super().reset()
