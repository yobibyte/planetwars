# PlanetWars State Representation
#
# written by Michael Buro and Graham Kendall Jan-27-2015
#
# - generate orders
# - execute single order for one player
# - execute orders for both players

import math
import random
from collections import defaultdict
from planetwars import Fleet, Planet, Order
from planetwars.utils import partition

class State:

  # extract relevant state data from global planetwars object
  def __init__(self, planets=None, fleets=None):
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

  # generate random start state
  # point symmetric planets
  # (no ships in the air)
  # fixme: planets may overlap in webview
  def random_setup(self, p1_planets, p2_planets, neutral_planets):

    WH     = 25.0  # Width/Height
    MARGIN = 1.0   # Margin

    self.planets = []
    self.fleets = []

    pmax = max(p1_planets, p2_planets)

    if neutral_planets % 2 == 1:
      # odd: create neutral planet at center
      self.add_planet(MID, MID, 0, self.rnd_ships(), self.rnd_growth())
      neutral_planets -= 1

    # create pmax+neutral_planet/2 symmetrical planet pairs
    # with neutral excess planets
    id = 0
    for i in range(pmax+neutral_planets/2):
      x = self.rnd_coord(WH, MARGIN)
      y = self.rnd_coord(WH, MARGIN)
      s = self.rnd_ships()
      g = self.rnd_growth()
      p1 = 1 if i < p1_planets else 0
      p2 = 2 if i < p2_planets else 0
      self.add_planet(x, y,       p1, s, g, id)
      id += 1
      self.add_planet(WH-x, WH-y, p2, s, g, id)
      id += 1

  def add_planet(self, x, y, owner, ships, growth, id):
    self.planets.append(Planet(id, x, y, owner, ships, growth))
     
  def rnd_coord(self, WH, MARGIN):
    return random.random()*(WH-2*MARGIN)+MARGIN

  def rnd_ships(self):
    return random.randint(5, 100)

  def rnd_growth(self):
    return random.randint(1, 5)
