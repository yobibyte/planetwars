import numpy
import random
import math

from .. import planetwars_ai
from planetwars.datatypes import Order
from planetwars.utils import *
from ..state import State

def dist(x1, y1, x2, y2):
  return math.sqrt((x1-x2)*(x1-x2)+(y1-y2)*(y1-y2))


# evaluate state (after executing action)
# pairwise features (source i, target j)
#   
def select_move(pid, planets, fleets):

  assert pid == 1 or pid == 2, "what?"

  my_planets, other_planets = partition(lambda x: x.owner == pid, planets)
  your_planets, neutral_planets = partition(lambda x: x.owner != 0, other_planets)

  wv = [
    +1.0,  # src ships
    -2.1,  # dest ships
    +0.0,  # my ships total
    +0.0,  # your ships total
    +0.0,  # neutral ship total
    +0.0,  # my total growth
    +0.0,  # your total growth
    -0.9,  # distance
    +0.0,  # I own dst planet
    +9.0,  # you own dst planet
    +5.0,  # neutral own dst planet    
    -0.5,  # src growth
    +1.0,  # dst growth
  ]

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
  
  for p in planets:
    if p.id == 0:
      neutral_ships_total += p.ships
    elif p.id == pid:
      my_growth += p.growth
      my_ships_total += p.ships
    else:
      your_ships_total += p.ships
      your_growth += p.growth

  if len(my_planets) == 0:
    return []

  # compute maximal distance between planets
  max_dist = 0
  for src in planets:
    for dst in planets:
      d = dist(src.x, dst.x, src.y, dst.y)
      if d > max_dist:
        max_dist = d

  # incoming ship bucket matrix
  # incoming friendly ships count 1, incoming enemy ships count -1
  # for each planet we tally incoming ship for time buckets 0..buckets-1
  # where buckets refers to the maximum distance on the map
  tally = numpy.zeros((len(planets), buckets))

  for f in fleets:
    # d = remaining distance
    d = dist(planets[f.source].x, planets[f.destination].x, planets[f.source].y, planets[f.destination].y) * \
        (f.remaining_turns/f.total_turns)
    b = d/max_dist * buckets
    if b >= buckets:
      b = buckets-1
    tally[f.destination, b] += f.ships * (1 if f.owner == pid else -1)
  
  best_sum = float("-inf")

  for src in my_planets:
    for dst in planets:
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
      fv.append(dist(src.x, dst.x, src.y, dst.y))

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
      if sum > best_sum:
        best_sum = sum
        best_order = Order(src, dst, src.ships/2)
        
  return [best_order]

@planetwars_ai("AgentTest")
def agenttest_ai(turn, pid, planets, fleets):

    # my_planets, other_planets = partition(lambda x: x.owner == pid, planets)
    # state  = State(planets, fleets)
    # orders = state.generate_orders(pid)

    # for p in planets:
    #   print "planet: ", p.id, p.x, p.y, p.owner, p.ships, p.growth
      
    # for f in fleets:
    #   print "fleet: ", f.owner, f.ships, f.source, f.destination, f.total_turns, f.remaining_turns

    # for o in orders:
    #   print "order:", o.source.id, o.destination.id, o.ships

    return select_move(pid, planets, fleets)
    
    # returned_orders = [random.choice(orders)]
    # print "RETURNED: ", returned_orders
    # return returned_orders
