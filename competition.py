import sys
import random
import argparse

from planetwars import PlanetWars
from planetwars.views import TextView

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--collisions', action='store_true', required=False, default=False,
                        help="Should the ships collide among each other?")
    parser.add_argument('--games', type=int, required=False, default=1000,
                        help="Number of turns per second run by the game.")
    arguments, remaining = parser.parse_known_args(argv)

    for i in range(arguments.games):
        game = PlanetWars(remaining[:2], map_name="map%i" % random.randint(1,100), collisions=arguments.collisions)
        game.play()
        sys.stdout.write('.')

if __name__ == '__main__':
    import sys
    main(sys.argv[1:])
