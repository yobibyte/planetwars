from planetwars import PlanetWars

def planetwars_ai(name):
    def planetwars_ai_decorator(f):
        PlanetWars.ais[name] = f
        return f
    return planetwars_ai_decorator

def planetwars_class(original):
    PlanetWars.ais[original.__name__] = original()
    return original

from .bots import *
