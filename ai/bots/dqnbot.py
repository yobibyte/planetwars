import numpy as np
import random
import math

from .. import planetwars_class
from planetwars.datatypes import Order
from planetwars.utils import *

from keras.models import Sequential                                             
from keras.layers import Dense, Activation

# TODO store only the best src,dst features in the state
# TODO change input dim to 3d 
# TODO use fleet for net inpt

@planetwars_class
class DQN(object):

    mem_size=10000
    memory = []
    counter=0
    n_planets = 23
    
    input_dim = n_planets*5
    output_dim = n_planets**2
    model = Sequential()
    model.add(Dense(100, batch_input_shape=(None, input_dim)))
    model.add(Activation('relu'))
    model.add(Dense(100))
    model.add(Activation('relu'))
    model.add(Dense(output_dim))
    model.add(Activation('sigmoid'))
    model.compile(loss='mse', optimizer='rmsprop', metrics=['accuracy'])
    # model.load_weights("model.h5")

    def __init__(self, eps=0.1, gamma=0.98, bsize=32): 
      self.last_state = None
      self.last_action = None
      self.eps = eps
      self.gamma = gamma
      self.n_features = 33
      self.bsize = bsize


    def update_memory(self, new_state, reward, terminal):
      if len(DQN.memory)<DQN.mem_size:
        DQN.memory.append([self.last_state, self.last_action, new_state, reward, terminal])
      else:
        del DQN.memory[0]
        DQN.memory.append([self.last_state, self.last_action, new_state, reward, terminal])
        self.train()

    def make_smart_move(self, planets, fleets):
      features = self.state_to_features((planets, fleets))  
      scores = DQN.model.predict(np.array([features]))
      move_idx = np.argmax(scores)
      s_i, d_i = np.unravel_index(move_idx, (n_planets, n_planets))
      return planets[s_i], planets[d_i]  
    
    def state_to_features(self,state):
      features = []
      for p in state[0]:
        features.append(p.x)
        features.append(p.y)
        features.append(p.owner)
        features.append(p.ships)
        features.append(p.growth)
      return np.array(features)

    def make_random_move(self, src, dst):
      source = random.choice(src)
      destination = random.choice(dst)
      return source, destination

    def train(self):

      idx = np.random.randint(0, DQN.mem_size, size=self.bsize)
      sampled_states = [DQN.memory[i] for i in idx]
      
      Y = np.zeros((self.bsize, 23*23))
      inpt = np.array([self.state_to_features(s[2]) for s in sampled_states])
      preds = np.max(DQN.model.predict(inpt), axis=1)
      for i, m_idx in enumerate(idx):
        a_idx = np.ravel_multi_index(DQN.memory[m_idx][1], (23,23))
        if DQN.memory[m_idx][4]:
          Y[i][a_idx] = DQN.memory[m_idx][3]
        else:
          Y[i][a_idx] = DQN.memory[m_idx][3] + self.gamma*preds[i]
      
      DQN.model.train_on_batch(np.array([self.state_to_features(s[0]) for s in sampled_states]), Y)
      
      DQN.counter+=1
      if DQN.counter==10000:
        #DQN.model.save_weights("model.h5", overwrite=True)
        DQN.counter=0

    def __call__(self, turn, pid, planets, fleets):
        self.pid = pid
        self.turn = turn

  
        my_planets, other_planets = partition(lambda x: x.owner == pid, planets)
        if len(my_planets) == 0 or len(other_planets) == 0:
          return []
               
        self.last_state = (planets, fleets) 

        if len(DQN.memory)<DQN.mem_size or random.random() < self.eps:
          my_planets, other_planets = partition(lambda x: x.owner == pid, planets)
          src, dst = self.make_random_move(my_planets, other_planets)
        else:
          src, dst = self.make_smart_move(planets,fleets)

        #self.eps *= 0.98
        self.last_action = (src.id, dst.id)
        
        return [Order(src, dst, src.ships/2)]

    def done(self, won, turns):
        pass
