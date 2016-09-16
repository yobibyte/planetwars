import numpy as np
import random
import math

from .. import planetwars_class
from planetwars.datatypes import Order
from planetwars.utils import *

from keras.models import Sequential                                             
from keras.layers import Dense, Activation

# TODO store only the best src,dst features in the state

@planetwars_class
class DQN(object):

    mem_size=10000
    memory = []
    counter=0
    
    input_dim = 33
    output_dim = 1

    model = Sequential()
    model.add(Dense(30, batch_input_shape=(None, input_dim)))
    model.add(Activation('relu'))
    model.add(Dense(output_dim))
    model.add(Activation('sigmoid'))
    model.compile(loss='mse', optimizer='sgd', metrics=['accuracy'])
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


    def make_features(self,src,dst,my_ships_total,your_ships_total,neutral_ships_total, my_growth,your_growth,buckets,tally, pid):

      fv = []
      fv.append(src.ships)
      fv.append(dst.ships)
      fv.append(my_ships_total)
      fv.append(your_ships_total)
      fv.append(neutral_ships_total)
      fv.append(my_growth)
      fv.append(your_growth)
      fv.append(math.sqrt((src.x - dst.x)**2 + (src.y - dst.y)**2))
      fv.append(1 if dst.id == pid else 0)
      fv.append(1 if dst.id != 0 and dst.id != pid else 0)
      fv.append(1 if dst.id == 0 else 0)
      fv.append(src.growth)
      fv.append(dst.growth)
      for i in range(buckets):
        fv.append(tally[src.id, i])
      for i in range(buckets):
        fv.append(tally[dst.id, i])

      return np.asarray([fv])

    def make_smart_move(self, planets, fleets, turn, pid):
      general_features = make_state_features(turn, pid, planets, fleets)
      features = []
      for s in srcs:
        for d in dsts:
          features.append(self.make_features(s,d, *general_features))
      scores = DQN.model.predict(np.array(features))
      move_idx = np.argmax(scores)
      s_i, d_i = np.unravel_index(move_idx, (len(s), len(d)))
      return s[s_i], d[d_i]  

    def make_state_features(self, turn, pid, planets, fleets):

      my_planets, other_planets = partition(lambda x: x.owner == pid, planets)
      your_planets, neutral_planets = partition(lambda x: x.owner != 0, other_planets)

      buckets = 10
      my_ships_total = 0
      your_ships_total = 0
      neutral_ships_total = 0
      my_growth = 0
      your_growth = 0
      
      for p in planets:
        if p.id == 0:
          neutral_ships_total += p.ships
        elif p.id == pid:
          my_growth += p.growth
          my_ships_total += p.ships
        else:
          your_ships_total += p.ships
          your_growth += p.growth

      max_dist = 0
      for src in planets:
        for dst in planets:
          d = math.sqrt((src.x - dst.x)**2 + (src.y - dst.y)**2)
          if d > max_dist:
            max_dist = d

      tally = np.zeros((len(planets), buckets))

      for f in fleets:
        d = dist(planets[f.source], planets[f.destination]) * \
            (f.remaining_turns/f.total_turns)
        b = d/max_dist * buckets
        if b >= buckets:
          b = buckets-1
        tally[f.destination, b] += f.ships * (1 if f.owner == pid else -1)

      return my_ships_total, your_ships_total, neutral_ships_total, my_growth, your_growth, buckets, tally


    def make_random_move(self, src, dst):
      source = random.choice(src)
      destination = random.choice(dst)
      return source, destination

    def Q_approx(sampled):
      preds = []
      sf = make_state_features()
      general_features = make_state_features(turn, pid, planets, fleets)

      for y in sampled:
        features = []
        for s in srcs:
          for d in dsts:
            features.append(self.make_features(s,d, *general_features))
        preds.append(np.max(Q.predict(features)))

      return preds

    def train(self):

      inputs = np.zeros((min(len(DQN.memory), self.bsize), DQN.input_dim))
      targets = np.zeros((inputs.shape[0], DQN.output_dim))

      idx = np.random.randint(0, len(DQN.memory), size=self.bsize)
      sampled_states = np.array(DQN.memory[i] for i in idx)
      Y = np.array([DQN.memory[i][3] for i in idx])
      preds = Q_approx(np.array([ss[2] for ss in sampled_states]))
      for i, m_idx in enumerate(idx):
        if not DQN.memory[m_idx][4]:
          Y[i] = Y[i]+self.gamma*preds[i]
 
      DQN.model.train_on_batch(np.array(ss[1] for ss in sampled_states),Y)
      DQN.counter+=1
      if counter==10000:
        DQN.model.save_weights("model.h5", overwrite=True)
        DQN.counter=0
        print counter

    def __call__(self, turn, pid, planets, fleets):

        if len(my_planets) == 0 or len(other_planets) == 0:
          return []
               
        self.last_state = (planets, fleets) 

        if not len(DQN.memory)<DQN.mem_size or random.random() < self.eps:
          my_planets, other_planets = partition(lambda x: x.owner == pid, planets)
          src, dst = self.make_random_move(my_planets, other_planets)

        else:
          src, dst = self.move(turn,planets,fleets, self.last_state)

        #self.eps *= 0.98
        self.last_action = (src, dst)
        
        return [Order(src, dst, src.ships/2)]

    def done(self, won, turns):
        pass
