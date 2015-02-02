import sys
import random
import argparse

from planetwars import PlanetWars
from planetwars.views import TextView
from ai.state import State

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--collisions', action='store_true', required=False, default=False,
                        help="Should the ships collide with each other?")
    parser.add_argument('--games', type=int, required=False, default=1000,
                        help="Number of turns per second run by the game.")
    parser.add_argument('--seed', type=int, required=False, default=0,
                        help="Initial rng seed, 0 = system-based")
    parser.add_argument('--p1num', type=int, required=False, default=1,
                        help="Planet number for player 1.")
    parser.add_argument('--p2num', type=int, required=False, default=1,
                        help="Planet number for player 2.")
    parser.add_argument('--nnum', type=int, required=False, default=10,
                        help="Number of neutral planets.")
    parser.add_argument('--genmaps', action='store_true', required=False, default=False,
                        help="Generate random maps.")
    parser.add_argument('--quiet', action='store_true', required=False, default=False,
                        help="Suppress all output to the console.")

    arguments, remaining = parser.parse_known_args(argv)

    seed = 0
    if arguments.seed == 0:
      # use system seed and print the resulting random integer
      seed = random.randint(1, 2000000000)
    else:
      # use passed on seed
      seed = arguments.seed

    random.seed(seed)
    print "seed=", seed  #, "rnd1=", random.randint(1, 2000000000)

    if arguments.genmaps:
      print "p1num=", arguments.p1num
      print "p2num=", arguments.p2num
      print "nnum=",  arguments.nnum

    sums = [0, 0, 0]
    
    for i in range(arguments.games):
      if arguments.genmaps:
        state = State()
        state.random_setup(arguments.p1num, arguments.p2num, arguments.nnum)
        game = PlanetWars(remaining[:2], planets=state.planets, fleets=state.fleets, collisions=arguments.collisions)
      else:
        game = PlanetWars(remaining[:2], map_name="map%i" % random.randint(32,32), collisions=arguments.collisions)
      winner, ship_counts = game.play()
      # print "game ", i, " winner = ", winner
      sums [winner] += 1
    print sums

      #sys.stdout.write('.')
      #sys.stdout.flush()

    print sums

if __name__ == '__main__':
    import sys
    main(sys.argv[1:])
