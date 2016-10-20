def reward_function1(pid, p_before, p_after, f_before=None, f_after=None):
  '''You get +1 for conquering new planet
     and -1 for losing one
  '''
  n_p = len(p_before)
  reward = 0.0
  for i in range(n_p):
    if (p_before[i].owner == pid and p_after[i].owner != pid):
      reward-=1.0*p_before[i].growth
    elif(p_before[i].owner != pid and p_after[i].owner ==pid):
      reward+=1.0*p_before[i].growth
  return reward



def reward_function2(pid, p_before, p_after, f_before, f_after, turn):

  mg=ms=mga=msa=0
  
  mg = sum([plt.growth for plt in p_before if plt.owner==pid])
  ms = sum([plt.ships for plt in p_before if plt.owner==pid])
  ms += sum([f.ships for f in f_before if f.owner==pid])
  yg = sum([plt.growth for plt in p_before if plt.owner!=pid and plt.owner!=0])
  ys = sum([plt.ships for plt in p_before if plt.owner!=pid and plt.owner!=0])
  ys += sum([f.ships for f in f_before if f.owner!=pid])
  mg /= float(yg+mg)
  ms /= float(ys+ms)

  mga = sum([plt.growth for plt in p_after if plt.owner==pid])
  msa = sum([plt.ships for plt in p_after if plt.owner==pid])
  msa += sum([f.ships for f in f_after if f.owner==pid])
  yga = sum([plt.growth for plt in p_after if plt.owner!=pid and plt.owner!=0])
  ysa = sum([plt.ships for plt in p_after if plt.owner!=pid and plt.owner!=0])
  ysa += sum([f.ships for f in f_after if f.owner!=pid])
  mga /= float(yga+mga)
  msa /= float(ysa+msa)

  return mga*(1-turn/200.0)+msa - (mg*(1-(turn-1)/200.0)+ms)


