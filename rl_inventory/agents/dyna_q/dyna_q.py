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
    def __init__(self):
        pass
    def select_action(state_key):
        pass
    def update(s, a, reward, s_next, terminated):
        pass
    def _q_learning_update(s, a, reward, s_next, terminated):
        pass
    def _planning_updates(self):
        pass