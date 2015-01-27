import random

from .. import planetwars_ai
from planetwars.datatypes import Order
from planetwars.utils import *
from ..state import State

@planetwars_ai("AgentTest")
def agenttest_ai(turn, pid, planets, fleets):

    my_planets, other_planets = partition(lambda x: x.owner == pid, planets)

    state  = State(planets, fleets)
    orders = state.generate_orders(pid)

    for p in planets:
      print "planet: ", p.id, p.x, p.y, p.owner, p.ships, p.growth
      
    for f in fleets:
      print "fleet: ", f.owner, f.ships, f.source, f.destination, f.total_turns, f.remaining_turns

    for o in orders:
      print "order:", o.source.id, o.destination.id, o.ships

    returned_orders = [random.choice(orders)]
    print "RETURNED: ", returned_orders
    return returned_orders
