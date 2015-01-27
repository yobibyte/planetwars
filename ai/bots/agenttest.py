import random

from .. import planetwars_ai
from planetwars.datatypes import Order
from planetwars.utils import *
from ..state import State

@planetwars_ai("AgentTest")
def agenttest_ai(turn, pid, planets, fleets):
    def mine(x):
        return x.owner == pid
    my_planets, other_planets = partition(mine, planets)
    source = random.choice(my_planets)
    destination = random.choice(other_planets)

    state = State(planets, fleets)

    orders = state.generate_orders(pid)

    for p in planets:
      print "planet: ", p.id, p.x, p.y, p.owner, p.ships, p.growth
      
    for f in fleets:
      print "fleet: ", f.owner, f.ships, f.source.id, f.destination.id, f.total_turns, f.remaining_turns
    
    for o in orders:
      print "order:", o.source.id, o.destination.id, o.source.ships

    return [Order(source, destination, source.ships / 2)]
