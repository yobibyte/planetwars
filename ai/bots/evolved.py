# evolved single-neuron agent 
# written by Michael Buro and Simon Lucas, Jan-27-2015
# with the help of Alex Champandard
#
# evolved by Simon Lucas

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
  
  buckets = 10

  # evolved from manually tuned parameters (agenttest.py)
  # much better!
  wv = [0.86775324593161407, -1.9010099177760116, 0.70433224708478814,
   0.38139294338347268, 0.54156215783780215, -0.26850400844879657,
   0.036186433238020135, -2.4578285185718509, -0.22894880101333545,
   8.6048706141753843, 5.9310406603617754, -1.7418715081600573,
   1.7733665060956196, 0.75099526122482629, -0.24516344621926955,
   -0.58245534282527744, -1.9683307387551237, -4.5097072412766952,
   -3.3414734358820031, -1.6702149156291219, -2.8390520001774457,
   -5.380093894197751, -5.3042483457349077, -1.4015940841792449,
   -0.8879621041490966, -2.3146041377067901, -0.75674029604322557,
   -2.194825427869195, -2.7245477291943003, -3.5719201691901739,
   -3.9302814137729709, -3.6712154408545712, -5.9446914488922378]
  

  # Friday Morning:
  # wv = [0.87228527101416398, -0.79126636923666871,
  #       -0.34114544122826324, -1.2090512361564403, 1.4031361661285386,
  #       0.73674878406987754, -0.516640269495791, -3.1994083525469721,
  #       -0.5745394282397579, -0.22981889285018697, -1.417569495893277,
  #       -2.259612849650309, -1.417122273379628, 0.22398736570314653,
  #       1.1025017800623469, -2.3399564135382556, 0.46038305190135953,
  #       -0.19313043611744043, 0.83489567169438406, -0.52030771008719445,
  #       1.5391963665964257, -0.87786985834090447, 1.2020126747923154,
  #       0.72289829744304113, -1.2814151831330289, 1.3404938004129179,
  #       -0.58430476934010023, -2.2650232722412449, -0.47874802835808361,
  #       -1.0517218313694681, -0.1941961917394518, -1.110827613720589,
  #       -1.543892566692554]


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
      # print "Agent2: ", src.id, dst.id, sum

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
  # print "AgentTest: ", best_order
  return [best_order]

@planetwars_ai("Evolved")
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
