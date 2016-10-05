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

    mem_size=10000
    memory = []

    counter = 0
    Q_v = 0
    Q_v_ctr = 0
    
    input_dim = 24
    output_dim = 1

    model = Sequential()
    model.add(Dense(100, batch_input_shape=(None, input_dim)))
    model.add(Activation('relu'))
    model.add(Dense(100))
    model.add(Activation('relu'))
    model.add(Dense(output_dim))
    model.add(Activation('linear'))
    opt = RMSprop(lr=0.0025)
    model.compile(loss='mse', optimizer=opt, metrics=['accuracy'])

    # model.load_weights("model.h5")
    # model.load_weights("model_pretrained_with_random.h5")


    selftrain = False
    tips = 0.0
    eps = 1


    def __init__(self, gamma=0.9, bsize=32): 
      self.last_state = None
      self.last_action = None
      self.gamma = gamma
      self.bsize = bsize
      
    def get_memory(self, selftrain):
      return DQN.memory if not selftrain else self.memory

    def get_model(self, selftrain):
      return DQN.model if not selftrain else self.model

    def switch(self):
      self.pid = 1 if self.pid==2 else 2


    def update_memory(self, last_state, action, new_state, reward, terminal, r_id):

      if self.pid!=r_id:
        self.switch()
        general_features = self.make_state_features(*last_state)
        self.last_state = self.make_features(action[0],action[1], *general_features)

      self.get_memory(DQN.selftrain).append([self.last_state, (new_state, r_id), reward, terminal])

      if len(self.get_memory(DQN.selftrain)) > DQN.mem_size:
        del self.get_memory(DQN.selftrain)[0]
        self.train()


    # def update_memory_twice(self, last_state, action, new_state, reward, reward_e, terminal):

    #   DQN.memory.append([self.last_state, (new_state, 2), reward, terminal])
    #   if len(DQN.memory) > DQN.mem_size:
    #     del DQN.memory[0]
    #     self.train()


    #   self.pid = 1
    #   general_features = self.make_state_features(*last_state)
    #   self.last_state = self.make_features(action[0],action[1], *general_features)

    #   DQN.memory.append([self.last_state, (new_state, 1), reward_e, terminal])
    #   if len(DQN.memory) > DQN.mem_size:
    #     del DQN.memory[0]
    #     self.train()

    # def update_memory_dd(self, *args):
    #   for i in range(len(args)):
    #     self.update_memory(*args[i])

    def local_model(self):
      self.memory = []

      self.model = Sequential()
      self.model.add(Dense(100, batch_input_shape=(None, DQN.input_dim)))
      self.model.add(Activation('relu'))
      self.model.add(Dense(100))
      self.model.add(Activation('relu'))
      self.model.add(Dense(DQN.output_dim))
      self.model.add(Activation('linear'))
      self.opt = RMSprop(lr=0.0025)
      self.model.compile(loss='mse', optimizer=self.opt, metrics=['accuracy'])
      self.model.load_weights("model.h5")
      
      DQN.selftrain = True


    def make_state_features(self, planets, fleets):

      total_ships=0
      total_growth=0
      total_fleets=0

      buckets = 10

      my_ships_total = 0
      your_ships_total = 0
      neutral_ships_total = 0
      my_growth = 0
      your_growth = 0

      for p in planets:
        if p.owner == 0:
          neutral_ships_total += p.ships
        elif p.owner == self.pid:
          # my_growth += p.growth
          my_ships_total += p.ships
        else:
          your_ships_total += p.ships
          # your_growth += p.growth

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
      # my_ships_total /= float(total_ships)
      # your_ships_total /= float(total_ships)
      # neutral_ships_total /= float(total_ships)
      # my_growth /= float(total_growth)
      # your_growth /= float(total_growth)

      # return total_ships, total_growth, my_ships_total, your_ships_total, neutral_ships_total, my_growth, your_growth, buckets, tally
      return buckets, tally


    # def make_features(self, src, dst, total_ships, total_growth, my_ships_total,your_ships_total,neutral_ships_total, my_growth,your_growth,buckets,tally):

    def make_features(self, src, dst, buckets,tally):

      # fv = []

      # if not(src==dst):      
      #   fv.append(src.ships/float(total_ships))
      #   fv.append(dst.ships/float(total_ships))
      #   fv.append(my_ships_total)
      #   fv.append(your_ships_total)
      #   fv.append(neutral_ships_total)
      #   fv.append(my_growth)
      #   fv.append(your_growth)
      #   fv.append(dist(src, dst))
      #   fv.append(1 if dst.owner == self.pid else 0)
      #   fv.append(1 if dst.owner != 0 and dst.owner != self.pid else 0)
      #   fv.append(1 if dst.owner == 0 else 0)
      #   fv.append(src.growth/float(total_growth))
      #   fv.append(dst.growth/float(total_growth))
      #   for i in range(buckets):
      #     fv.append(tally[src.id, i])
      #   for i in range(buckets):
      #     fv.append(tally[dst.id, i])

      # else:
      #   fv.append(0)
      #   fv.append(0)
      #   fv.append(my_ships_total)
      #   fv.append(your_ships_total)
      #   fv.append(neutral_ships_total)
      #   fv.append(my_growth)
      #   fv.append(your_growth)
      #   fv.append(0)
      #   fv.append(0)
      #   fv.append(0)
      #   fv.append(0)
      #   fv.append(0)
      #   fv.append(0)
      #   for i in range(buckets):
      #     fv.append(0)
      #   for i in range(buckets):
      #     fv.append(0)

      # return fv

      fv = []

      if not(src==dst):      
        fv.append(1 if src.ships>dst.ships+dst.growth*dist(src,dst)*2 else 0)
        fv.append(1 if dst.owner == self.pid else 0)
        fv.append(1 if dst.owner != 0 and dst.owner != self.pid else 0)
        fv.append(1 if dst.owner == 0 else 0)
        for i in range(buckets):
          fv.append(tally[src.id, i])
        for i in range(buckets):
          fv.append(tally[dst.id, i])

      else:
        fv.append(0)
        fv.append(0)
        fv.append(0)
        fv.append(0)
        for i in range(buckets):
          fv.append(0)
        for i in range(buckets):
          fv.append(0)

      return fv


    def make_smart_move(self, planets, fleets):

      general_features = self.make_state_features(planets, fleets)
      features = []
      srcs, _ = partition(lambda x: x.owner == self.pid, planets)
      for s in srcs:
        for d in planets:
          features.append(self.make_features(s,d, *general_features))
 

      scores = self.get_model(DQN.selftrain).predict(np.array(features))

      
      move_idx = np.argmax(scores)
      self.last_state = features[move_idx]
      s_i, d_i = np.unravel_index(move_idx, (len(srcs), len(planets)))

      DQN.Q_v += np.max(scores)
      DQN.Q_v_ctr += 1

      return srcs[s_i], planets[d_i]  


    def make_random_move(self, planets, fleets):

      s_f = self.make_state_features(planets, fleets)

      # if random.random()<DQN.tips:
      #   my_planets, theirs, neutral = aggro_partition(self.pid, planets)

      #   DQN.tips *= 0.9999
      #   src = max(my_planets, key=get_ships)

      #   e_dst = [e_plt.growth/dist(src,e_plt)/((e_plt.ships-np.sum(s_f[-1][e_plt.id])) if e_plt.ships!=np.sum(s_f[-1][e_plt.id]) else 0.1) for e_plt in theirs]
      #   n_dst = [n_plt.growth/dist(src,n_plt)/((n_plt.ships-abs(np.sum(s_f[-1][n_plt.id]))) if n_plt.ships!=abs(np.sum(s_f[-1][n_plt.id])) else 0.1) for n_plt in neutral]
      #   e_dst.append(-np.inf)
      #   n_dst.append(-np.inf)

      #   dst = theirs[np.argmax(e_dst)] if np.max(e_dst)>=np.max(n_dst) else neutral[np.argmax(n_dst)]

      # else:
      my_planets, other_planets = partition(lambda x: x.owner == self.pid, planets)
      src = random.choice(my_planets)
      dst = random.choice(other_planets)

      self.last_state = self.make_features(src, dst, *s_f)
      return src, dst
    
    

    def Q_approx(self, sampled):
      
      sp_idx = []
      features = []
      for y in sampled:
        if self.pid != y[1]:
          self.switch()
        
        general_features = self.make_state_features(y[0][0], y[0][1])
        srcs, _ = partition(lambda x: x.owner == self.pid, y[0][0])
        features.append(self.make_features(None, None, *general_features))
        for s in srcs:
          for d in y[0][0]:
            features.append(self.make_features(s, d, *general_features))
            
        c_i = len(srcs)*len(y[0][0])+1
        if(len(sp_idx)!=0):
          c_i += sp_idx[-1]
        sp_idx.append(c_i)  

      features = np.array(features)
      preds = np.split(self.get_model(DQN.selftrain).predict(features), sp_idx[:-1])
      return np.array([np.max(r) for r in preds])


    def train(self):
      #DQN.memory.append([self.last_state, new_state, reward, terminal])
      idx = np.random.randint(0, len(self.get_memory(DQN.selftrain)), size=self.bsize)
      sampled_states = sorted([self.get_memory(DQN.selftrain)[i] for i in idx], key=lambda s: s[3])
      terms = np.array([s[3] for s in sampled_states])

      n_nonterms = self.bsize - np.sum(terms)
      Y = np.array([s[2] for s in sampled_states])    
      Y[:n_nonterms] += self.gamma*self.Q_approx([s[1] for s in sampled_states[:n_nonterms]])  

      X = np.zeros((self.bsize, DQN.input_dim))
      for i,s in enumerate(sampled_states):
        X[i] = np.array(s[0])
      self.get_model(DQN.selftrain).train_on_batch(X, Y)
      DQN.counter += 1

    def __call__(self, turn, pid, planets, fleets):
        self.pid = pid
        self.turn = turn
  
        my_planets, other_planets = partition(lambda x: x.owner == pid, planets)
        if len(my_planets) == 0 or len(other_planets) == 0:
          return []

        if random.random()<DQN.eps:
          src, dst = self.make_random_move(planets,fleets)
          DQN.eps *= 0.9999 if DQN.eps>0.01 else 1
        else:
          src, dst = self.make_smart_move(planets,fleets)

        if src==dst:
          return []

        return [Order(src, dst, src.ships/2)]

    def done(self, won, turns):
        pass

    def save_weights(self):
        self.get_model(DQN.selftrain).save_weights("model.h5", overwrite=True)

    def load_weights(self):
        print 'load the model.............'
        self.get_model(DQN.selftrain).load_weights("model.h5")
        # self.get_model(DQN.selftrain).load_weights("model_pretrained_with_random.h5")

    def reset_Q(self):
        DQN.Q_v = DQN.Q_v_ctr = DQN.counter = 0








