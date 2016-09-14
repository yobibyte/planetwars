import numpy as np
import random

from .. import planetwars_class
from planetwars.datatypes import Order
from planetwars.utils import *

@planetwars_class
class DQN(object):
    
    def __init__(self, mem_size=10000, eps=0.1, memory=None, model=None):
      self.ctr = 0
      if memory is None:
        self.memory = [0 for i in range(mem_size)]
        self.is_mem_full = False
      else:
        self.is_mem_full = len(memory) == mem_size
      self.last_state = None
      self.last_action = None
      self.eps = eps

    def update_memory(self):
      pass
  
    def make_smart_move(self, src, dst):
      pass

    def train(self):
      # sample random minibatch of transitions
      # compute targets y's like here: https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf
      # perform a gradient step
      pass

    def __call__(self, turn, pid, planets, fleets):
        
        def mine(x):
            return x.owner == pid
        
        my_planets, other_planets = partition(mine, planets)

        if len(my_planets) == 0 or len(other_planets) == 0:
          return []
        
        if not self.is_mem_full or random() < self.eps:
          source = random.choice(my_planets)
          destination = random.choice(other_planets)
        else:
          self.make_smart_move(my_planets, other_planets)        
        
        self.train()        

        return [Order(source, destination, source.ships / 2)]

    def done(self, won, turns):
        pass
