import numpy as np
import random
import math

from .. import planetwars_class
from planetwars.datatypes import Order
from planetwars.utils import *

from keras.models import Sequential                                             
from keras.layers import Dense
from keras.layers.core import Dropout
from keras.optimizers import RMSprop


@planetwars_class
class DQN(object):

    Q_v = Q_v_ctr = counter = 0

    buckets = 5
    mem_size=10000
    bsize = 32
    memory = []
    gamma = 0.9
    eps   = 0.1
    model = Sequential()
    model.add(Dense(256, batch_input_shape=(None, 18), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='linear'))
    opt = RMSprop(lr=0.00025)
    model.compile(loss='mse', optimizer='rmsprop', metrics=['accuracy'])
    #model.load_weights("model.h5")

    def __init__(self): 
      self.last_state = None
    
    def update_memory(self, new_state, reward, terminal):
      DQN.memory.append([self.last_state, new_state, reward, terminal])
      if len(DQN.memory) > DQN.mem_size:
        del DQN.memory[0]
        self.train()

    def make_state_features(self, planets, fleets):

      total_ships = np.sum([p.ships for p in planets])
      #total_growth = np.sum([p.growth for p in planets if p.owner!=0])
      total_growth = np.sum([p.growth for p in planets])
      max_dist = np.max([dist(src, dst) for src in planets for dst in planets])
      
      tally = np.zeros((len(planets), DQN.buckets))
      for f in fleets:
        total_ships += f.ships
        d = dist(planets[f.source], planets[f.destination]) * (f.remaining_turns/f.total_turns)
        b = d/max_dist * DQN.buckets
        if b >= DQN.buckets:
          b = DQN.buckets-1
        tally[f.destination, b] += f.ships * (1 if f.owner == self.pid else -1)
      tally /= float(total_ships)

      return total_ships, total_growth, tally, max_dist


    def make_features(self, src, dst, total_ships, total_growth, tally, max_dist):
      
      fv = [src.ships/float(total_ships), dst.ships/float(total_ships)]
      fv.append(dist(src, dst)/float(max_dist))
      fv.append(1 * (dst.owner == self.pid))
      fv.append(1 * (dst.owner != 0 and dst.owner != self.pid))
      fv.append(1 * (dst.owner == 0))
      fv.append(src.growth/float(total_growth))
      fv.append(dst.growth/float(total_growth))
      for i in range(DQN.buckets):
        fv.append(tally[src.id, i])
      for i in range(DQN.buckets):
        fv.append(tally[dst.id, i])
      return fv


    def make_smart_move(self, planets, fleets, sf):
      srcs = [p for p in planets if p.owner == self.pid]
      features = [self.make_features(s, d, *sf) for s in srcs for d in planets]
      scores = DQN.model.predict(np.array(features))
      move_idx = np.argmax(scores)
      s_i, d_i = np.unravel_index(move_idx, (len(srcs), len(planets)))
      DQN.Q_v += scores[move_idx]
      DQN.Q_v_ctr += 1
      return srcs[s_i], planets[d_i]  


    def Q_approx(self, sampled):
      res = np.zeros(len(sampled))
      for i, y in enumerate(sampled):
        sf = self.make_state_features(y[0],y[1])
        srcs = [p for p in y[0] if p.owner == self.pid]
        if len(srcs)!=0:
          features = [self.make_features(s, d, *sf) for s in srcs for d in y[0]]
        else:
          features = [[0]*(8+2*DQN.buckets)] 
        res[i] = np.max(DQN.model.predict(np.array(features)))

      return res

    def train(self):
      idx = np.random.randint(0, len(DQN.memory), size=DQN.bsize)
      ss = sorted([DQN.memory[i] for i in idx], key=lambda s: s[3])
      X = np.array([s[0] for s in ss])
      Y = np.array([s[2] for s in ss]) 
      n_nonterms = DQN.bsize - np.sum([s[3] for s in ss])
      Y[:n_nonterms] += DQN.gamma*self.Q_approx([s[1] for s in ss[:n_nonterms]])  
      DQN.model.train_on_batch(X, Y)
      DQN.counter += 1

    def __call__(self, turn, pid, planets, fleets):
        self.pid = pid
        my_planets, other_planets = partition(lambda x: x.owner == pid, planets)
        
        sf = self.make_state_features(planets, fleets) 
        self.last_state = [0]*(8+2*DQN.buckets) 

        if len(my_planets) == 0 or len(other_planets) == 0:
          return []

        if random.random()<self.eps or len(DQN.memory)<DQN.mem_size:
          src, dst = random.choice(my_planets), random.choice(planets)
        else:
          src, dst = self.make_smart_move(planets,fleets, sf)
        
        # if the state was not 'no source'
        self.last_state = self.make_features(src, dst, *sf) 

        if src == dst:
          return []
        return [Order(src, dst, src.ships/2)]


    def done(self, won, turns):
        pass

    def save_weights(self):
        DQN.model.save_weights("model.h5", overwrite=True)

    def load_weights(self):
        DQN.model.load_weights("model.h5")

    def reset_Q(self):
      DQN.Q_v = DQN.Q_v_ctr = DQN.counter = 0
      
