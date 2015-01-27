import random

from .. import planetwars_ai
from planetwars.datatypes import Order
from planetwars.utils import *

@planetwars_ai("Random2")
def random2_ai(turn, pid, planets, fleets):
    def mine(x):
        return x.owner == pid
    my_planets, other_planets = partition(mine, planets)
    source = random.choice(my_planets)
    destination = random.choice(other_planets)
    return [Order(source, destination, source.ships / 2)]
