def reward_function1(pid, p_before, p_after, f_before=None, f_after=None):
  '''You get +1 for conquering new planet
     and -1 for losing one
  '''
  n_p = len(p_before)
  reward = 0.0
  for i in range(n_p):
    if (p_before[i].owner == pid and p_after[i].owner != pid):
      reward-=1.0
    elif(p_before[i].owner != pid and p_after[i].owner ==pid):
      reward+=1.0
  return reward



def reward_function2(pid, p_before, p_after, f_before, f_after):
    # compute the increasing of ships
  
  reward = sum([plt.ships for plt in p_after if plt.owner==pid])
  reward += sum([f.ships for f in f_after if f.owner==pid])    
  # reward -= sum([plt.ships for plt in p_before if plt.owner==pid])
  # reward -= sum([f.ships for f in f_before if f.owner==pid])


  return float(reward)
