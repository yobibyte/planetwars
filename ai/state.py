# PlanetWars State Representation
#
# - generate orders
# - execute single orders

from collections import defaultdict
from planetwars import Fleet, Planet, Order
from planetwars.utils import partition

class State:

  # extract relevant state data from global planetwars object
  def __init__(self, planets, fleets):
    self.planets, self.fleets = planets, fleets;

  # generate list of all possible order for player pid
  def generate_orders(self, pid):
    def mine(x):
      return x.owner == pid
    my_planets, other_planets = partition(mine, self.planets)
    
    res = []
     
    for src in my_planets:
      for dst in other_planets:
        res.append(Order(src, dst, src.ships / 2))

    return res

  # NOT TESTED YET
  # execute a single order for player with id pid
  def execute_order(self, pid, order):

    # Departure
    self.issue_order(pid, order)

    # copied from game.py
    
    # Advancement
    for planet in self.planets:
      planet.generate_ships()

    for fleet in self.fleets:
      fleet.advance()
        
    # Arrival
    arrived_fleets, self.fleets = partition(lambda fleet: fleet.has_arrived(), self.fleets)
    for planet in self.planets:
      planet.battle([fleet for fleet in arrived_fleets if fleet.destination == planet])

  # NOT TESTED YET
  # execute orders for both players
  def execute_order(self, pid1, orders1, pid2, orders2):

    assert pid1 != pid2, "pids equal"
    
    # Departures
    for o1 in orders1:
      self.issue_order(pid1, o1)

    for o2 in orders2:
      self.issue_order(pid2, o2)
      
    # copied from game.py

    # Advancement
    for planet in self.planets:
      planet.generate_ships()

    for fleet in self.fleets:
      fleet.advance()
        
    # Arrival
    arrived_fleets, self.fleets = partition(lambda fleet: fleet.has_arrived(), self.fleets)
    for planet in self.planets:
      planet.battle([fleet for fleet in arrived_fleets if fleet.destination == planet])

  # NOT TESTED YET
  # copied from game.py
  def issue_order(self, player, order):   # player?  id or 
    if order.source.owner != player:
      raise Exception("Player %d issued an order from enemy planet %d." % (player, order.source.id))
    source = order.source # !!! was: self.planets[order.source.id]
    ships = int(min(order.ships, source.ships))
    if ships > 0:
      destination = order.destination # !!! was: self.planets[order.destination.id]
      source.ships -= ships
      self.fleets.append(Fleet(player, ships, source, destination))

