import time
from collections import defaultdict

from planetwars import Fleet, Planet
from planetwars.internal import load_all_maps
from planetwars.utils import count_ships, partition

class State:

  def __init__(self, planetwars):
    self.planets, self.fleets = planetwars.freeze();

  def generate_actions(self, pid):
        def mine(x):
          return x.owner == pid
        my_planets, other_planets = partition(mine, planets)

        
        source = random.choice(my_planets)
        destination = random.choice(other_planets)
        return [Order(source, destination, source.ships / 2)]
# implement

  def execute_order(self, player, order):
# implement
    
