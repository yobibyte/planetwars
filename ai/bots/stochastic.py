import random

from .. import planetwars_class
from planetwars.datatypes import Order
from planetwars.utils import *

from .sample import all_to_close_or_weak, strong_to_weak

@planetwars_class
class Stochastic(object):

    EPSILON = 1.0

    def __call__(self, turn, pid, planets, fleets):
        # if random.random() < Stochastic.EPSILON:
        #    return strong_to_weak(turn, pid, planets, fleets)

        my_planets, other_planets = partition(lambda x: x.owner == pid, planets)
        if len(my_planets) == 0 or len(other_planets) == 0:
          return []

        source = random.choice(my_planets)
        if random.random() > Stochastic.EPSILON:
            destination = random.choice(other_planets)
        else:
            destination = random.choice(planets)
        return [Order(source, destination, source.ships * 0.5)]

    def done(self, *args):
        pass
