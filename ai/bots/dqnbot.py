import numpy as np
import random

from .. import planetwars_class
from planetwars.datatypes import Order
from planetwars.utils import *

from keras.models import Sequential                                             
from keras.layers import Dense, Activation

@planetwars_class
class DQN(object):

    def __init__(self, mem_size=10000, eps=0.1, gamma=0.98, bsize=32, memory=None, model=None):
      self.mem_size = mem_size
      if memory is None:
        self.memory = [0 for i in range(mem_size)]
        self.is_mem_full = False
        self.ctr = 0
      else:
        self.memory = memory
        self.is_mem_full = len(memory) == mem_size
        if self.is_mem_full:
          ctr = 0
        else:
          ctr = len(memory)
      
      self.last_state = None
      self.last_action = None
      self.eps = eps
      self.gamma = gamma
      self.n_features = 33
      self.bsize = bsize
      if model:
        self.model = model
      else:
        self.init_model()
  
    def init_model(self):
      self.model = Sequential()
      self.model.add(Dense(10, batch_input_shape=(None, 23*6)))
      self.model.add(Activation('relu'))
      self.model.add(Dense(23*23))
      self.model.add(Activation('relu'))

    def update_memory(self, new_state, reward, terminal):
      # state is a tuple (planets, fleet)
      # reward is scalar
      # terminal is boolean
      self.memory[self.ctr] = (self.last_state, self.last_action, new_state, reward, terminal)
      self.ctr+=1
      if self.ctr == self.mem_size:
        self.ctr = 0

    def get_memory(self):
      return self.memory

    def set_memory(self, memory):
      self.memory = memory

    def planets2state(self, planets):
      res = []
      for p in planets:
        res.append(p.id) 
        res.append(p.x) 
        res.append(p.y) 
        res.append(p.owner) 
        res.append(p.ships)
        res.append(p.growth) 
      return np.array(res)
 
    def make_random_move(self, src, dst):
      source = random.choice(src)
      destination = random.choice(dst)
      return source, destination

    def make_smart_move(self, src, dst):
      return self.make_random_move(src, dst)

    def compute_features(src, dest):
      pass

    def train(self):
      batch_indices = np.random.choice(self.memory, self.bsize)
      # change to SEQUENCE FUNCTION phi, k=4 as in paper
      X = np.array(self.planets2state(self.memory[idx][0]) for idx in batch_indices)
      Y = np.zeros(self.bsize)
      for i, state in enumerate(X):
        if state[-1]:
          Y[i] = state[-2]
        else:
          #features = np.array(len(src)*len(dest), self.n_features)
          #for s,s_p in enumerate(src):
          #  for d,d_p in enumerate(dest):
          #    features[s,p] = get_features(src,dest) 
          #scores = net.predict(features)
          #take maximum best score from prediction
          #future_rewards = 0
          #Y[i] = state[-2]+self.gamma*future_rewards
          Y[i] = max(net.predict(X[i]))
      self.model.fit(X,Y, batch_size=self.bsize, nb_epoch=1)     

    def __call__(self, turn, pid, planets, fleets):

        def mine(x):
            return x.owner == pid
        
        my_planets, other_planets = partition(mine, planets)

        if len(my_planets) == 0 or len(other_planets) == 0:
          return []
        
        if not self.is_mem_full or random() < self.eps:
          src, dest = self.make_random_move(my_planets, other_planets)  
        else:
          src, dest = self.make_smart_move(my_planets, other_planets)        
          self.train()        
        
        self.last_state  = (planets, fleets)
        self.last_action = (src, dest) 

        return [Order(src, dest, src.ships/2)]

    def done(self, won, turns):
        pass
