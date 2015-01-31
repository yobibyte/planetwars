import random

from .. import planetwars_class
from planetwars.datatypes import Order
from planetwars.utils import *

from .sample import all_to_close_or_weak

@planetwars_class
class Stochastic(object):

    def __call__(self, turn, pid, planets, fleets):
        # if random.random() > 0.99:
        #    return all_to_close_or_weak(turn, pid, planets, fleets)

        def mine(x):
            return x.owner == pid

        my_planets, other_planets = partition(mine, planets)
        if len(my_planets) == 0 or len(other_planets) == 0:
          return []

        source = random.choice(my_planets)
        destination = random.choice(other_planets)
        return [Order(source, destination, source.ships / 2)]

    def done(self, won, turns):
        pass
