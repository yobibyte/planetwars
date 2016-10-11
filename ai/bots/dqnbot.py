import numpy as np
import random
import math

from .. import planetwars_class
from planetwars.datatypes import Order
from planetwars.utils import *

from keras.models import Sequential                                             
from keras.layers import Dense, Activation
from keras.optimizers import RMSprop


@planetwars_class
class DQN(object):

    counter = 0
    Q_v = 0
    Q_v_ctr = 0

    mem_size=10000
    memory = []
    model = Sequential()
    model.add(Dense(64, batch_input_shape=(None, 19)))
    model.add(Activation('relu'))
    model.add(Dense(1))
    model.add(Activation('linear'))
    opt = RMSprop(lr=0.01)
    model.compile(loss='mse', optimizer='rmsprop', metrics=['accuracy'])
    model.load_weights("model.h5")

    def __init__(self, eps=0.1, gamma=0.9, bsize=32): 
      self.last_state = None
      self.eps = eps
      self.gamma = gamma
      self.bsize = bsize
    
    def update_memory(self, new_state, reward, terminal):
      DQN.memory.append([self.last_state, new_state, reward, terminal])
      if len(DQN.memory) > DQN.mem_size:
        del DQN.memory[0]
        self.train()

    def make_state_features(self, planets, fleets):

      total_ships=0
      total_growth=0
      total_fleets=0

      buckets = 3
      my_ships_total = 0
      your_ships_total = 0
      neutral_ships_total = 0
      my_growth = 0
      your_growth = 0

      for p in planets:
        if p.owner == 0:
          neutral_ships_total += p.ships
        elif p.owner == self.pid:
          my_growth += p.growth
          my_ships_total += p.ships
        else:
          your_ships_total += p.ships
          your_growth += p.growth

      max_dist = 0
      for src in planets:
        for dst in planets:
          d = dist(src, dst)
          if d > max_dist:
            max_dist = d

      tally = np.zeros((len(planets), buckets))
      for f in fleets:
        total_fleets += f.ships
        d = dist(planets[f.source], planets[f.destination]) * (f.remaining_turns/f.total_turns)
        b = d/max_dist * buckets
        if b >= buckets:
          b = buckets-1
        tally[f.destination, b] += f.ships * (1 if f.owner == self.pid else -1)

      total_ships = total_fleets+my_ships_total+your_ships_total+neutral_ships_total
      total_growth = my_growth+your_growth

      tally /= float(total_ships)
      my_ships_total /= float(total_ships)
      your_ships_total /= float(total_ships)
      neutral_ships_total /= float(total_ships)
      my_growth /= float(total_growth)
      your_growth /= float(total_growth)

      return total_ships, total_growth, my_ships_total, your_ships_total, neutral_ships_total, my_growth, your_growth, buckets, tally


    def make_features(self, src, dst, total_ships, total_growth, my_ships_total,your_ships_total,neutral_ships_total, my_growth,your_growth,buckets,tally):

      fv = []
      fv.append(my_ships_total)
      fv.append(your_ships_total)
      fv.append(neutral_ships_total)
      fv.append(my_growth)
      fv.append(your_growth)
      if not src is None:
        fv.append(src.ships/float(total_ships))
        fv.append(dst.ships/float(total_ships))
        fv.append(dist(src, dst))
        fv.append(1 if dst.owner == self.pid else 0)
        fv.append(1 if dst.owner != 0 and dst.owner != self.pid else 0)
        fv.append(1 if dst.owner == 0 else 0)
        fv.append(src.growth/float(total_growth))
        fv.append(dst.growth/float(total_growth))
        for i in range(buckets):
          fv.append(tally[src.id, i])
        for i in range(buckets):
          fv.append(tally[dst.id, i])
      else:
        fv.extend([0]*(8+2*buckets))

      return fv


    def make_smart_move(self, planets, fleets):

      general_features = self.make_state_features(planets, fleets)
      features = []
      srcs, _ = partition(lambda x: x.owner == self.pid, planets)
      for s in srcs:
        for d in planets:
          features.append(self.make_features(s,d, *general_features))
 
      scores = DQN.model.predict(np.array(features))
      move_idx = np.argmax(scores)
      self.last_state = features[move_idx]
      s_i, d_i = np.unravel_index(move_idx, (len(srcs), len(planets)))

      DQN.Q_v += np.max(scores)
      DQN.Q_v_ctr += 1
      return srcs[s_i], planets[d_i]  


    def make_random_move(self, planets, fleets):
      mine, _ = partition(lambda x: x.owner == self.pid, planets)
      return random.choice(mine), random.choice(planets)
    
    def Q_approx(self, sampled):
      sp_idx = []
      features = []
      for y in sampled:
        general_features = self.make_state_features(y[0], y[1])
        srcs, _ = partition(lambda x: x.owner == self.pid, y[0])
        features.append(self.make_features(None, None, *general_features))
        for s in srcs:
          for d in y[0]:
            features.append(self.make_features(s, d, *general_features))
            
        c_i = len(srcs)*len(y[0])+1
        if(len(sp_idx)!=0):
          c_i += sp_idx[-1]
        sp_idx.append(c_i)  

      features = np.array(features)
      preds = np.split(DQN.model.predict(features), sp_idx[:-1])
      return np.array([np.max(r) for r in preds])


    def train(self):
      idx = np.random.randint(0, len(DQN.memory), size=self.bsize)
      sampled_states = sorted([DQN.memory[i] for i in idx], key=lambda s: s[3])
      X = np.array([s[0] for s in sampled_states])
      Y = np.array([s[2] for s in sampled_states])    
      terms = np.array([s[3] for s in sampled_states])
      n_nonterms = self.bsize - np.sum(terms)
      Y[:n_nonterms] += self.gamma*self.Q_approx([s[1] for s in sampled_states[:n_nonterms]])  
      DQN.model.train_on_batch(X, Y)
      DQN.counter += 1

    def __call__(self, turn, pid, planets, fleets):
        self.pid = pid
        self.turn = turn
        my_planets, other_planets = partition(lambda x: x.owner == pid, planets)

        if len(my_planets) == 0 or len(other_planets) == 0:
          return []

        if random.random()<self.eps or len(DQN.memory)<DQN.mem_size:
          src, dst = self.make_random_move(planets,fleets)
        else:
          src, dst = self.make_smart_move(planets,fleets)

        if src==dst:
          return []
        
        ls_gen_features = self.make_state_features(planets, fleets) 
        self.last_state = self.make_features(src, dst, *ls_gen_features) 

        return [Order(src, dst, src.ships/2)]

    def done(self, won, turns):
        pass

    def save_weights(self):
        DQN.model.save_weights("model.h5", overwrite=True)

    def load_weights(self):
        DQN.model.load_weights("model.h5")

    def reset_Q(self):
      DQN.Q_v = DQN.Q_v_ctr = DQN.counter = 0
      
