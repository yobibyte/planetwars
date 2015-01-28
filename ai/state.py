# PlanetWars State Representation
#
# written by Michael Buro and Graham Kendall Jan-27-2015
#
# - generate orders
# - execute single order for one player
# - execute orders for both players

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

  # generate random start state
  # point symmetric planets
  # (no ships in the air)
  # todo: get in charge of rng seed to make runs repeatable
  # (should also apply
  def random_setup(p1_planets, p2_planets, neutral_planets):

    WH     = 25.0  # Width/Height
    MARGIN = 1.0   # Margin

    self.planets = []
    self.fleets = []

    pmax = math.max(p1_players, p2_players)

    if neutral_planets % 2 == 1:
      # odd: create neutral planet at center
      add_planet(MID, MID, 0, rnd_ships(), rnd_growth())
      neutral_planets--

    # create pmax+neutral_planet/2 symmetrical planet pairs
    # with neutral excess planets
    id = 0
    for i in range(pmax+neutral_planets/2):
      x = rnd_coord(WH, MARGIN)
      y = rnd_coord(WH, MARGIN)
      s = rnd_ships()
      g = rnd_growth()
      p1 = 1 if i < p1_planets else 0
      p2 = 2 if i < p2_planets else 0
      add_planet(x, y,       p1, s, g, id++)
      add_planet(WH-x, WH-y, p2, s, g, id++)

   def add_planet(x, y, owner, ships, growth, id):
     self.planets.append(ImmutablePlanet(id, x, y, ownwer, ships, growth))
     
   def rnd_coord(WH, MARGIN):
     return random.random()*(WH-2*MARGIN)+MARGIN

   def rnd_ships():
     return random.randint(5, 100)

   def rnd_growth():
     return random.randint(1, 5)
