# single-neuron agent using trained weights
# written by Michael Buro

import numpy as np
import random
import math

from .. import planetwars_class
from planetwars.datatypes import Order
from planetwars.utils import *
from ..state import State
from ..draft_interface import bot_act
from ..bots.nn.deepq.deepq import DeepQ
from ..bots.nn.sknn.sknn import IncrementalMinMaxScaler


# Euclidean distance
def dist(src, dst):
  dx = src.x - dst.x
  dy = src.y - dst.y
  d = math.sqrt(dx*dx + dy*dy)
  return d

np.set_printoptions(threshold='nan')

@planetwars_class
class DeepBot(object):

  def __init__(self):
    layers  =  [("RectifiedLinear", 15), ("Linear", )]
    self.avg_reward = 0
    self.games = 0
    self.scaler = IncrementalMinMaxScaler()
    try:
        self.bot = DeepQ.load()
        print "Loaded"
    except:
        self.bot = DeepQ(layers)
        print "Not loaded"

  def __call__(self, turn, pid, planets, fleets):

    assert pid == 1 or pid == 2, "what?"

    my_planets, other_planets = partition(lambda x: x.owner == pid, planets)
    your_planets, neutral_planets = partition(lambda x: x.owner != 0, other_planets)

  # create feature matrix
  # [  feature_list for move 1, ... , feature_list for move n ]
  #
    if len(my_planets) == 0 or len(other_planets) == 0:
      return []

  #  for i,p in enumerate(planets):
  #    print "PLANET: ", i, p.id, "x=", p.x, "y=", p.y, p.owner, p.ships
    
    my_ships_total = 0
    your_ships_total = 0
    neutral_ships_total = 0
    my_growth = 0
    your_growth = 0

    # tally ships and growth
    
    for p in planets:
      if p.id == 0:
        neutral_ships_total += p.ships
      elif p.id == pid:
        my_growth += p.growth
        my_ships_total += p.ships
      else:
        your_ships_total += p.ships
        your_growth += p.growth

    # compute maximal distance between planets

    max_dist = 0
    for src in planets:
      for dst in planets:
        d = dist(src, dst)
        if d > max_dist:
          max_dist = d

    # incoming ship buckets
    buckets = 10
          
    # incoming ship bucket matrix
    # incoming friendly ships count 1, incoming enemy ships count -1
    # for each planet we tally incoming ship for time buckets 0..buckets-1
    # where buckets refers to the maximum distance on the map
    tally = np.zeros((len(planets), buckets))

    for f in fleets:
      # d = remaining distance
      d = dist(planets[f.source], planets[f.destination]) * \
          (f.remaining_turns/f.total_turns)
      b = d/max_dist * buckets
      if b >= buckets:
        b = buckets-1
      tally[f.destination, b] += f.ships * (1 if f.owner == pid else -1)
    
    all_orders = []
    fm = []
    
    for src in my_planets:
      for dst in planets:
        # allowing NOOPs
        #if src == dst:
        #  continue

        fv = []
        # planet totals
        fv.append(src.ships)
        fv.append(dst.ships)

        # ship total
        fv.append(my_ships_total)
        fv.append(your_ships_total)
        fv.append(neutral_ships_total)

        # growth
        fv.append(my_growth)
        fv.append(your_growth)

        # distance
        d = dist(src, dst)
        #print "PLANET DIST: ", src, dst
        #print "DIST: ", src.id, dst.id, d
        fv.append(d)

        # I own dst planet
        fv.append(1.0 if dst.id == pid else 0.0)
        # you own dst planet
        fv.append(1.0 if dst.id != 0 and dst.id != pid else 0.0)
        # neutral owns dst planet
        fv.append(1.0 if dst.id == 0 else 0.0)

        # growth
        fv.append(src.growth)
        fv.append(dst.growth)

        # incoming ship buckets (src)

        # print "incoming src", src.id, ": ", 

        for i in range(buckets):
          fv.append(tally[src.id, i])
          # print i, tally[src.id, i],
        #print
        
        # incoming ship buckets (dst)
        # print "incoming dst", dst.id, ": ", 

        for i in range(buckets):
          fv.append(tally[dst.id, i])
          #print tally[dst.id, i],
        #print

        # todo: add more percentage options
        # need to create one feature vector for each option

        perc = 50 # ship percentage
        # fv.append(perc)  
        
        fm.append(fv);
        all_orders.append(Order(src, dst, src.ships*perc/100))

    # intermediate reward = 0 for now      

    order_ids = bot_act(fm, 0)
    
    npfm = np.array(fm)
    npfm += np.array([0., 0., 0., 118., 0., 5., 43., 0., 0., 0., 0., 0., 0., -257., -320., -338., -288., -502., -329., -213., -129., -106., -106., -381.
, -381., -381., -458., -502., -329., -219., -129., -106., -106.])
    npfm /= np.array([6.2700E+02, 6.2700E+02, 6.2700E+02, 2.8440E+03, 5.7900E+02, 0.0000E+00, 5.0000E+01, 3.31519109E+01, 1.0, 1.0, 1.0,\
                      5.0000E+00, 5.0000E+00, 5.7900E+02, 6.8700E+02, 6.3600E+02, 6.0100E+02, 8.1500E+02, 4.7200E+02, 3.4500E+02, 2.7200E+02, 3.0500E+02,\
                      2.3800E+02, 7.0300E+02, 7.4800E+02, 7.4200E+02, 8.1900E+02, 8.1500E+02, 6.0300E+02, 4.2200E+02, 2.9700E+02, 3.0500E+02, 2.5000E+02])
    # self.scaler.fit(npfm)
    npfm = npfm.clip(0.0, 1.0)
    # print npfm
    bestord_id = self.bot.act(npfm,0,0)
    #self.bot.fit(0, 0, npfm)
    order_ids = [ bestord_id]
    orders = []
    for id in order_ids:
      orders.append(all_orders[id])

    return orders

  # inform learner that game ended
  def done(self, won):
    self.games += 1
    self.avg_reward += float(won) - 0.5
    #self.bot.fit(self.bot.last_sa, float(won), 1, None)
    self.bot.addToMemory(self.bot.last_sa, float(won), 1, None)
    self.bot.last_sa = None
    if self.games % 10 == 0:
      print '#', int(self.games), "(%i)" % len(self.bot.memory), self.avg_reward/self.games*2
      print "Training...",      
      self.bot.train_from_memory(1000)
      print "DONE!"
      self.avg_reward = 0.0
      self.games = 0
      #self.bot.save()

    #print 'after', int(self.games), self.avg_reward/self.games*2
    #if(self.games % 1 == 0 ):
    #self.avg_reward = 0

    # print self.scaler.data_min
    # print self.scaler.data_max - self.scaler.data_min
