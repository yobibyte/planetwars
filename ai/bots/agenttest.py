# single-neuron agent with manually "tuned" weights
# written by Michael Buro and Simon Lucas, Jan-27-2015
# with the help of Alex Champandard

import numpy
import random
import math

from .. import planetwars_ai
from planetwars.datatypes import Order
from planetwars.utils import *
from ..state import State

# Euclidean distance
def dist(src, dst):
  dx = src.x - dst.x
  dy = src.y - dst.y
  d = math.sqrt(dx*dx + dy*dy)
  return d


# evaluate state (after executing action)
# pairwise features (source i, target j)
#   
def select_move(pid, planets, fleets):

  assert pid == 1 or pid == 2, "what?"

  my_planets, other_planets = partition(lambda x: x.owner == pid, planets)
  if len(my_planets) == 0 or len(other_planets) == 0:
    return []

  your_planets, neutral_planets = partition(lambda x: x.owner != 0, other_planets)

#  for i,p in enumerate(planets):
#    print "PLANET: ", i, p.id, "x=", p.x, "y=", p.y, p.owner, p.ships
  
  # weights
  wv = [
    +1.0,  # src ships
    -2.1,  # dest ships
    +0.0,  # my ships total
    +0.0,  # your ships total
    +0.0,  # neutral ship total
    +0.0,  # my total growth
    +0.0,  # your total growth
    -1.5,  # distance , was -1.5
    +0.0,  # I own dst planet
    +9.0,  # you own dst planet
    +5.0,  # neutral own dst planet    
    -0.5,  # src growth
    +1.0,  # dst growth
  ]

  # incoming ship buckets
  buckets = 10

  # src incoming ship buckets
  for i in range(buckets):
    wv.append(-i*0.5)

  # dst incoming ship buckets
  for i in range(buckets):
    wv.append(-i*0.5)
  
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

  # incoming ship bucket matrix
  # incoming friendly ships count 1, incoming enemy ships count -1
  # for each planet we tally incoming ship for time buckets 0..buckets-1
  # where buckets refers to the maximum distance on the map
  tally = numpy.zeros((len(planets), buckets))

  for f in fleets:
    # d = remaining distance
    d = dist(planets[f.source], planets[f.destination]) * \
        (f.remaining_turns/f.total_turns)
    b = d/max_dist * buckets
    if b >= buckets:
      b = buckets-1
    tally[f.destination, b] += f.ships * (1 if f.owner == pid else -1)
  
  best_sum = float("-inf")
  best_orders = []
  
  for src in my_planets:
    for dst in planets:
      if src == dst:
        continue

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
      fv.append(1 if dst.id == pid else 0)
      # you own dst planet
      fv.append(1 if dst.id != 0 and dst.id != pid else 0)
      # neutral owns dst planet
      fv.append(1 if dst.id == 0 else 0)

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
      
      assert len(fv) == len(wv), "lengths disagree " + str(len(fv)) + " " + str(len(wv))

      # compute value (weights * features)
      sum = 0
      for i,f in enumerate(fv):
        sum += f*wv[i]

      # update best action (use tie-breaker?)
      if sum >= best_sum:
        if sum > best_sum:
          best_orders = []
        best_sum = sum
        best_orders.append(Order(src, dst, src.ships/2))

  # print "#ORDERS: ", len(best_orders),
  best_order = random.choice(best_orders)

  #if best_order.source == best_order.destination:
  #  print "SAME PLANET!"
  #print
  return [best_order]

@planetwars_ai("AgentTest")
def agenttest_ai(turn, pid, planets, fleets):
  return select_move(pid, planets, fleets)

# my_planets, other_planets = partition(lambda x: x.owner == pid, planets)
# state  = State(planets, fleets)
# orders = state.generate_orders(pid)

# for p in planets:
#   print "planet: ", p.id, p.x, p.y, p.owner, p.ships, p.growth

# for f in fleets:
#   print "fleet: ", f.owner, f.ships, f.source, f.destination, f.total_turns, f.remaining_turns

# for o in orders:
#   print "order:", o.source.id, o.destination.id, o.ships

# returned_orders = [random.choice(orders)]
# print "RETURNED: ", returned_orders
# return returned_orders
