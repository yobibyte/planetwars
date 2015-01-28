# call-back functions invoked by player
#
# returns list of integers that encode concurrent actions to be executed
# (e.g. all actions with values > max_value - epsilon)
# 
def bot_act(state_action_feature_vectors, transition_reward):
  pass

# is called when game ends
# won: true iff player won the game
def bot_done(won):
  pass
