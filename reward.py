def reward_function1(pid, p_before, p_after):
  '''You get +1 for conquering new planet
     and -1 for losing one
  '''
  n_p = len(p_before)
  reward = 0
  for i in range(n_p):
    if (p_before[i].owner == pid and p_after[i] != pid):
      reward-=1
    elif(p_before[i].owner != pid and p_after[i] ==pid):
      reward+=1
  return reward
