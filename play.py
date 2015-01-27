import argparse
from planetwars import PlanetWars
from planetwars.views import TextView

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--collisions', action='store_true', required=False, default=False,
                        help="Should the ships collide among each other?")
    parser.add_argument('--rate', type=int, required=False, default=100,
                        help="Number of turns per second run by the game.")
    parser.add_argument('--map', type=str, required=False, default="map1",
                        help="The filename without extension for planets.")
    parser.add_argument('--quiet', action='store_true', required=False, default=False,
                        help="Suppress all output to the console.")
    arguments, remaining = parser.parse_known_args(argv)

    game = PlanetWars(remaining[:2], map_name=arguments.map, turns_per_second=arguments.rate, collisions=arguments.collisions)
    game.add_view(TextView(arguments.quiet))
    game.play()

if __name__ == '__main__':
    import sys
    main(sys.argv[1:])
