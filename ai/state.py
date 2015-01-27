import time
from collections import defaultdict

from planetwars import Fleet, Planet
from planetwars.internal import load_all_maps
from planetwars.utils import count_ships, partition

class State:

  # extract relevant state data from global planetwars object
  def __init__(self, planetwars):
    self.planets, self.fleets = planetwars.freeze();

  # generate list of all possible order for player pid
  def generate_orders(self, pid):
    def mine(x):
      return x.owner == pid
    my_planets, other_planets = partition(mine, planets)
    
    res = []
    for src in my_planets:
      for dest in other_planets:
        res.extend(Order(src, dst, src.ships / 2))

    return res

  # execute a single order for player with id pid
  def execute_order(self, pid, order):

    # Departure
    self.issue_order(pid, order)
        
    # Advancement
    for planet in self.planets:
      planet.generate_ships()
      for fleet in self.fleets:
        fleet.advance()
        
    # Arrival
    arrived_fleets, self.fleets = partition(lambda fleet: fleet.has_arrived(), self.fleets)
    for planet in self.planets:
      planet.battle([fleet for fleet in arrived_fleets if fleet.destination == planet])

     
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
