###############################################################
# DYNA-Q DEMO SCRIPT
###############################################################
# This script demonstrates how to:
#   1. Create the environment
#   2. Create the StateDiscretizer
#   3. Train a DynaQAgent using real + simulated updates
#   4. Run a greedy demo episode with the trained agent
#
# This mirrors q_learning_demo.py but adds:
#   - planning_steps in the training loop
#   - faster learning because of model-based updates
###############################################################

from rl_inventory.envs.extended_inventory import ExtendedInventoryEnv
from rl_inventory.agents.qlearning.qlearning import StateDiscretizer
from rl_inventory.agents.dyna_q import DynaQAgent
import numpy as np

def train_agent(
    num_episodes: int = 200,
    max_steps_per_episode: int = 365,
    planning_steps: int = 10,
):
    ##########################################
    #   TRAIN DYNA Q AGENT
    ##########################################
    #   :param num_episodes:            How many episodes (years) to train for.
    #   :param max_steps_per_episode:   Maximum number of steps (days) per episode.
    #                                   Typically matches env.episode_length (365).
    #   :param planning_steps:          How many simulated planning updates the DynaQAgent performs after each real environment step.
    ##########################################
    # --------------------------------
    # (1) Create the environment and the StateDiscretizer.
    # --------------------------------
    env = ExtendedInventoryEnv(discrete_actions=True)
    disc = StateDiscretizer()
    # --------------------------------
    # (2) Create the Dyna Q agent
    # --------------------------------
    agent = DynaQAgent(
        n_actions=env.action_space.n,
        alpha=0.1,
        gamma=0.99,
        epsilon=0.2,
        planning_steps=planning_steps,
    )
    # --------------------------------
    # (3) Training loop over episodes
    # --------------------------------
    for episode in range(num_episodes):
        state = env.reset() # reset environment
        s = disc.discretize(state) # discretize the initial state
        total_reward = 0.0

        for _ in range(max_steps_per_episode):
            # ----------------------------------------
            # (3.1) Select action using epsilon-greedy
            # ----------------------------------------
            a = agent.select_action(s)
            # -----------------------------------------
            # (3.2) Take a real step in the environment
            # -----------------------------------------
            next_state, reward, terminated, truncated, info = env.step(a)
            # -----------------------------------------
            # (3.3) Discretize next state
            # -----------------------------------------
            s_next = disc.discretize(next_state)
            # ----------------------------------------
            # (3.4) Update Dyna-Q agent:
            #         - real Q-learning
            #         - model update
            #         - planning updates
            # ----------------------------------------
            agent.update(s, a, reward, s_next, terminated)
            s = s_next                  # move to next state
            total_reward += reward

            if terminated or truncated:
                break
    return agent, disc
def demo_run(agent: DynaQAgent, disc: StateDiscretizer):
    ###################################################################
    # RUN A SINGLE GREEDY DEMO EPISODE WITH THE TRAINED DYNA-Q AGENT
    ###################################################################
    #   :param agent:   A trained DynaQAgent instance.
    #   :param disc:    The same StateDiscretizer used during training.
    ###################################################################
    # -------------------------------------------
    # (1) Create the environment
    #--------------------------------------------
    env = ExtendedInventoryEnv(discrete_actions=True)
    # -------------------------------------------
    # (2) Reset and discretize initial state
    # -------------------------------------------
    state = env.reset()
    s = disc.discretize(state)

    total_reward = 0.0

    # -------------------------------------------
    # (3) Force GREEDY behavior (no exploration)
    # --------------------------------------------
    original_epsilon = agent.epsilon
    agent.epsilon = 0.0                 # always pick best action
    # --------------------------------------------
    # (4) Run one episode (day-by-day)
    # --------------------------------------------
    for day in range(env.episode_length):
        # ---------------------------------------------------------------
        # (4.1) Select GREEDY action (max Q-value for the current state)
        # ---------------------------------------------------------------
        q_vals = [agent.Q[(s, a)] for a in range(agent.n_actions)]
        a = int(np.argmax(q_vals))
        # --------------------------------------------------------------
        # (4.2) Take action in the environment
        # --------------------------------------------------------------
        next_state, reward, terminated, truncated, info = env.step(a)
        total_reward += reward
        # --------------------------------------------------------------
        # (4.3) Move to NEXT discretized state
        # --------------------------------------------------------------
        s = disc.discretize(next_state)

        if terminated or truncated:
            break
    # ------------------------------------------
    # (5) Restore epsilon + Show results
    # ------------------------------------------
    agent.epsilon = original_epsilon

    # Print total reward for the entire demo episode
    print(f"\n[Dyna-Q] Demo episode finished. Total reward = {total_reward:.2f}")
    # Plot the full episode history
    env.plot_episode()
def main():
    #######################
    # THE MAIN ENTRY POINT
    #######################
    # -------------------------------
    # (1) Train the Dyna-Q agent
    # -------------------------------
    agent, disc = train_agent(
        num_episodes=200,
        max_steps_per_episode=365,
        planning_steps=10,
    )
    # -------------------------------
    # (2) Run a greedy demo episode
    # -------------------------------
    demo_run(agent, disc)
if __name__== "__main__":
    main()