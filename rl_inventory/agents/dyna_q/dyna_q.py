###############################################################
# DYNA-Q AGENT
###############################################################
# Dyna-Q is a reinforcement learning algorithm that combines:
#   1. Real Q-learning updates (from actual environment steps)
#   2. A learned model of the environment (stores s,a -> r,s')
#   3. Planning updates using the model (imagined experience)
#
# EACH time the agent takes a step:
#
#   Step 1: The agent observes a real transition:
#              (s, a) -> reward r, next state s'
#
#   Step 2: The agent performs a normal Q-learning update
#
#   Step 3: The agent stores this transition in a MODEL:
#              Model[(s, a)] = (r, s', done_flag)
#
#   Step 4: The agent performs K PLANNING UPDATES:
#              - Sample a random (s,a) from the model
#              - Pretend to "replay" the experience
#              - Update Q-table again based on the replay
###############################################################

class DynaQAgent:
    def __init__(self, n_actions, alpha=0.1, gamma=0.99, epsilon=0.2, planning_steps=10):
        ############################################
        # Initialize all fields needed for Dyna-Q.
        ############################################
        #   :param n_actions:      number of possible actions in the environment
        #   :param alpha:          learning rate for Q-learning updates
        #   :param gamma:          discount factor
        #   :param epsilon:        exploration rate for epsilon-greedy action selection
        #   :param planning_steps: how many simulated updates to run per real step
        ############################################
        # ------------------------------------------
        # (1) STORE BASIC HYPERPARAMETERS
        # ------------------------------------------
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.planning_steps = planning_steps
        # ------------------------------------------
        # (2) Q-TABLE
        # ------------------------------------------
        # Q[(state_key, action)] = value
        from collections import defaultdict
        self.Q = defaultdict(float)
        # ------------------------------------------
        # (3) MODEL FOR DYNA-Q
        # ------------------------------------------
        # Model[(state_key, action)] = (reward, next_state_key, done_flag)
        #
        # Every time we take a real step in the environment,
        # we store that transition here.
        self.Model = {}

    def select_action(self, state_key):
        #################################################
        #  Choose an action using epsilon-greedy policy.
        #################################################
        #   :param state_key:   The discretized state representation used as a key in the Q-table.
        #################################################
        import numpy as np
        # ---------------------------------------------------------------
        # (1) Exploration: choose a random action
        # ---------------------------------------------------------------
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)
        # ---------------------------------------------------------------
        # (2) Exploitation: choose best Q-value action
        # ---------------------------------------------------------------
        q_vals = [self.Q[(state_key, a)] for a in range(self.n_actions)]
        best_action = int(np.argmax(q_vals))
        return best_action

    def update(self, s, a, reward, s_next, terminated):
        ################################################################
        # Perform the full Dyna-Q update for ONE real environment step.
        ################################################################
        #   :param s:          current state (discretized state key)
        #   :param a:          action the agent took in this state
        #   :param reward:     reward returned by the environment for (s,a)
        #   :param s_next:     next state (discretized state key) from env
        #   :param terminated: True if episode ended after this action
        ################################################################
        # ----------------------------------------------------------
        # (1) REAL EXPERIENCE UPDATE (normal Q-learning update)
        # ----------------------------------------------------------
        self._q_learning_update(s, a, reward, s_next, terminated)
        #-----------------------------------------------------------
        # (2) STORE THE TRANSITION IN THE MODEL FOR FUTURE PLANNING
        #-----------------------------------------------------------
        # Save for future simulated planning
        self.Model[(s, a)] = (reward, s_next, terminated)
        # ----------------------------------------------------------
        # (3) RUN PLANNING UPDATES
        # ----------------------------------------------------------
        self._planning_updates()
    def _q_learning_update(self, s, a, reward, s_next, terminated):
        ############################################################
        # APPLY THE Q-LEARNING UPDATE FORMULA
        ############################################################
        #   :param s:           The CURRENT state key (discretized) BEFORE taking action a.
        #   :param a:           The ACTION taken in state s.
        #   :param reward:      The immediate reward returned by the environment.
        #   :param s_next:      The NEXT state key (discretized) AFTER taking action a.
        #   :param terminated:  True if the episode ended after this transition.
        ############################################################
        # --------------------------------------------
        # Get the current Q-value for (s, a)
        # --------------------------------------------
        q_sa = self.Q[(s, a)]
        # --------------------------------------------
        # Case 1: Terminal state — no bootstrapping
        # --------------------------------------------
        if terminated:
            target = reward  # No future value because episode ends
        else:
            # --------------------------------------------
            # Case 2: Non-terminal — bootstrap using max_a' Q(s_next, a')
            # --------------------------------------------
            # Compute the maximum Q-value in the next state
            q_next_max = max(self.Q[(s_next, a_prime)] for a_prime in range(self.n_actions))
            target = reward + self.gamma * q_next_max
        # --------------------------------------------
        # Q-learning update:
        # Q(s,a) = Q(s,a) + alpha * (target - Q(s,a))
        # --------------------------------------------
        self.Q[(s, a)] = q_sa + self.alpha * (target - q_sa)
    def _planning_updates(self):
        ####################
        # PLANNING UPDATES
        ####################
        import random
        # ----------------------------------------------------------------------
        # If the model is empty, there is nothing to simulate
        # ----------------------------------------------------------------------
        if len(self.Model) == 0:
            return
        # ----------------------------------------------------------------------
        # Perform K planning updates
        # ----------------------------------------------------------------------
        for _ in range(self.planning_steps):
            # ------------------------------------------------------------------
            # (1) Sample a random (state, action) from the model
            # ------------------------------------------------------------------
            (s_m, a_m) = random.choice(list(self.Model.keys()))
            # ------------------------------------------------------------------
            # (2) Retrieve stored transition
            # ------------------------------------------------------------------
            reward_m, s_next_m, terminated_m = self.Model[(s_m, a_m)]
            # ------------------------------------------------------------------
            # (3) Apply simulated Q-learning update
            # ------------------------------------------------------------------
            self._q_learning_update(s_m, a_m, reward_m, s_next_m, terminated_m)