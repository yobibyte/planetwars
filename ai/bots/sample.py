import random

from .. import planetwars_ai
from planetwars.datatypes import Order
from planetwars.utils import *

@planetwars_ai("StrongToRandom")
def strong_to_random(turn, pid, planets, fleets):
    if random.random() > 0.5:
        return []

    my_planets, their_planets, neutral_planets = aggro_partition(pid, planets)
    other_planets = their_planets + neutral_planets
    if len(my_planets) == 0 or len(other_planets) == 0:
        return []
    my_strongest = max(my_planets, key=get_ships)
    other_random = random.choice(other_planets)
    return [Order(my_strongest, other_random, my_strongest.ships * 0.5)]

@planetwars_ai("StrongToWeak")
def strong_to_weak(turn, pid, planets, fleets):
    if random.random() > 0.75:
        return random_ai(turn, pid, planets, fleets)

    my_planets, their_planets, _ = aggro_partition(pid, planets)
    if len(my_planets) == 0 or len(their_planets) == 0:
      return []
    my_strongest = max(my_planets, key=get_ships)
    their_weakest = min(their_planets, key=lambda x: x.ships * 100 + turn_dist(my_strongest, x))
    return [Order(my_strongest, their_weakest, my_strongest.ships * 0.5)]

@planetwars_ai("StrongToClose")
def strong_to_close(turn, pid, planets, fleets):
    if random.random() > 0.75:
        return random_ai(turn, pid, planets, fleets)

    my_planets, their_planets, neutral_planets = aggro_partition(pid, planets)
    other_planets = their_planets + neutral_planets
    if len(my_planets) == 0 or len(other_planets) == 0:
        return []
    my_strongest = max(my_planets, key=get_ships)
    other_closest = min(other_planets, key=lambda x: turn_dist(my_strongest, x) * 1000 + x.ships)
    return [Order(my_strongest, other_closest, my_strongest.ships * 0.5)]

@planetwars_ai("StrongToBest")
def strong_to_best(turn, pid, planets, fleets):
    if random.random() > 0.5:
        return random_ai(turn, pid, planets, fleets)

    my_planets, their_planets, neutral_planets = aggro_partition(pid, planets)
    other_planets = their_planets + neutral_planets
    if len(my_planets) == 0 or len(other_planets) == 0:
        return []
    my_strongest = max(my_planets, key=get_ships)
    other_best = max(other_planets, key=lambda x: x.growth / (1.0+x.ships+turn_dist(my_strongest, x)*5))
    return [Order(my_strongest, other_best, my_strongest.ships * 0.5)]


@planetwars_ai("AllToWeak")
def all_to_weak(turn, pid, planets, fleets):
    my_planets, their_planets, _ = aggro_partition(pid, planets)
    if len(my_planets) == 0 or len(their_planets) == 0:
      return []
    destination = min(their_planets, key=get_ships)
    orders = []
    for planet in my_planets:
        orders.append(Order(planet, destination, planet.ships * 0.75))
    return orders

@planetwars_ai("AllToCloseOrWeak")
def all_to_close_or_weak(turn, pid, planets, fleets):
    my_planets, their_planets, neutral_planets = aggro_partition(pid, planets)
    if len(my_planets) == 0:
      return []
    other_planets = their_planets + neutral_planets
    if len(their_planets) == 0:
      return []
    their_weakest = min(their_planets, key=get_ships)
    my_total = sum(map(get_ships, my_planets))
    destination = min(their_planets, key=get_ships)
    orders = []
    if len(other_planets) == 0:
      return []
    for planet in my_planets:
        if random.random() < 0.5:
            def dist_to(other_planet):
                return turn_dist(planet, other_planet)
            closest = min(other_planets, key=dist_to)
            orders.append(Order(planet, closest, planet.ships * 0.75))
        else:
            orders.append(Order(planet, their_weakest, planet.ships * 0.75))
    return orders

@planetwars_ai("Random")
def random_ai(turn, pid, planets, fleets):
    def mine(x):
        return x.owner == pid
    my_planets, other_planets = partition(mine, planets)
    if len(my_planets) == 0 or len(other_planets) == 0:
      return []
    source = random.choice(my_planets)
    destination = random.choice(other_planets)
    return [Order(source, destination, source.ships / 2)]
