import time
from collections import defaultdict

from planetwars import Fleet, Planet
from planetwars.internal import load_all_maps
from planetwars.utils import count_ships, partition


def _neutral_player(turn, pid, planets, fleets):
    return []

class PlanetWars:

    ais = {}
    maps = load_all_maps()
    exploit = False
    epoch_ctr = 0
    DQNvsDQN = False

    def __init__(self, players, map_name=None, planets=None, fleets=None,
                                turns_per_second=None, turn=0, collisions=False):
        if len(players) < 2:
            raise Exception("A game requires at least two players.")
        self.player_names = players
        self.players = [_neutral_player] + [PlanetWars.ais[player] for player in players]
        self.map_name = map_name

        if map_name is not None:
            planets, fleets = PlanetWars.maps[map_name]
            self.planets = [Planet(*planet) for planet in planets]
            self.fleets = [Fleet(*fleet) for fleet in fleets]
        else:
            assert planets is not None, "Please specify planets since map_name is None."
            assert fleets is not None, "Please specify fleets since map_name is None."
            self.planets = planets
            self.fleets = fleets

        self.views = []
        self.turns_per_second = turns_per_second
        self.turn_duration = 1.0 / (turns_per_second or 1.0)
        self.turn = turn
        self.collisions = collisions


        for i, p in enumerate(self.players):
            if str(p).split()[0]=='<ai.bots.dqnbot.DQN':
                self.DQN_id=i
                self.DQN_player=p
            else:
                self.another_bot_id = i
        PlanetWars.DQNvsDQN = True if self.another_bot_id==0 else False
        self.another_bot_id = 1 if self.another_bot_id==0 else self.another_bot_id


        if self.DQN_player.counter>=10000:
            self.DQN_player.reset_Q()
            PlanetWars.epoch_ctr += 1
        

    def add_view(self, view):
        self.views.append(view)

    def freeze(self):
        planets = tuple(planet.freeze() for planet in self.planets)
        fleets = tuple(fleet.freeze() for fleet in self.fleets)
        return planets, fleets

    def play(self):

        dqnreward = dqnreward_e = 0
        reward = reward_e = 0

        planets, fleets = self.freeze()
        for view in self.views:
            view.initialize(self.turns_per_second, self.planets, self.map_name, self.player_names)
            view.update(planets, fleets)
        self.time_totals = [0 for i in range(len(self.players))]
        self.time_max    = [0 for i in range(len(self.players))]
        next_turn = time.time() + self.turn_duration
        winner = -1

        while winner < 0:
            # Wait until time has passed
            now = time.time()
            if self.turns_per_second is not None and now < next_turn:
                time.sleep(next_turn - now)
            next_turn += self.turn_duration
        
            # Do the turn
            reward, reward_e = self.do_turn()
            
            # Update views
            planets, fleets = self.freeze()
            for view in self.views:
                view.update(planets, fleets)
            # Check for end game.
            winner, ship_counts, turns = self.gameover()
            # print winner, ship_counts, turns

            dqnreward += reward
            dqnreward_e += reward_e

            if winner < 0:
                if not PlanetWars.DQNvsDQN:
                    if not PlanetWars.exploit:
                        self.DQN_player.update_memory(self.last_state, self.action, (planets, fleets), reward_e, False, self.another_bot_id)
                    else:
                        self.DQN_player.update_memory(self.last_state, self.action, (planets, fleets), reward, False, self.DQN_id)
                else:
                    self.DQN_player.update_memory_twice(self.last_state, self.action, (planets, fleets), reward, reward_e, False)
            else:
                if not PlanetWars.DQNvsDQN:
                    if not PlanetWars.exploit:
                        print "collect data from opponent"
                        self.DQN_player.update_memory(self.last_state, self.action, (planets, fleets), reward_e, True, self.another_bot_id)
                    else:
                        print "collect data from itself"
                        self.DQN_player.update_memory(self.last_state, self.action, (planets, fleets), reward, True, self.DQN_id)
                else:
                    self.DQN_player.update_memory_twice(self.last_state, self.action, (planets, fleets), reward, reward_e, True)



        for view in self.views:
            view.game_over(winner, ship_counts, turns)

        for i, p in enumerate(self.players):
            try:
                p.done(winner == i, turns)
            except AttributeError:
                pass

        return winner, ship_counts, turns, self.time_totals, self.time_max, dqnreward, dqnreward_e, self.DQN_player.counter, self.DQN_player.Q_v, self.DQN_player.Q_v_ctr

                
    def do_turn(self):
        """Performs a single turn of the game."""

        # Get orders
        planets, fleets = self.freeze()
        player_orders = []

        self.action = (None, None)
        self.last_state = (planets,fleets)


        for i, player in enumerate(self.players):
          prev = time.time()
          player_orders.append(player(self.turn, i, planets, fleets))
          t = time.time()-prev
          self.time_totals[i] += t
          if t > self.time_max[i]:
            self.time_max[i] = t

        score, score_e = self.compute_score(self.planets, self.fleets)

        self.turn += 1
                
        # Departure
        for player, orders in enumerate(player_orders):
#          print "turn", self.turn, "player:", player, "orders:", orders
          for order in orders:
            self.issue_order(player, order)


        # Advancement
        for planet in self.planets:
            planet.generate_ships()
        for fleet in self.fleets:
            fleet.advance()


        # Collisions do before advancement?
        if self.collisions:
            oldfleets = self.fleets
            for fleet in oldfleets:
                fleet.destroy = False
            self.fleets = []
            while len( oldfleets ) > 0:
                fleet = oldfleets.pop( 0 )
                if fleet.destroy:
                    continue
                fleetx, fleety = fleet.location()
                i = 0
                while i < len( oldfleets ):
                    checkx, checky = oldfleets[i].location()
                    if (fleetx - checkx)**2 + (fleety - checky)**2 < 1:
                        #print "DESTROY", fleetx, fleety, checkx, checky
                        oldfleets[i].destroy = True
                        fleet.destroy = True
                    i += 1
                if not fleet.destroy:
                    self.fleets.append( fleet )


        # Arrival
        arrived_fleets, self.fleets = partition(lambda fleet: fleet.has_arrived(), self.fleets)
        for planet in self.planets:
            planet.battle([fleet for fleet in arrived_fleets if fleet.destination == planet])

        next_score, next_score_e = self.compute_score(self.planets, self.fleets)


        if next_score - score > next_score_e - score_e:
            return 1.0, 0.0
        else:
            return 0.0, 1.0


    def compute_score(self, planets, fleets):
        
        score_s=0
        score_s = sum([plt.ships for plt in planets if plt.owner==self.DQN_id])
        score_s += sum([f.ships for f in fleets if f.owner==self.DQN_id])
        score_es = sum([plt.ships for plt in planets if plt.owner!=self.DQN_id and plt.owner!=0])
        score_es += sum([f.ships for f in fleets if f.owner!=self.DQN_id])
        
        return score_s, score_es


    def issue_order(self, player, order):
        if order.source.owner != player:
            raise Exception("Player %d issued an order from enemy planet %d." % (player, order.source.id))
        source = self.planets[order.source.id]
        ships = int(min(order.ships, source.ships))
        if ships > 0:
            destination = self.planets[order.destination.id]
            source.ships -= ships
            self.fleets.append(Fleet(player, ships, source, destination))

            if player!=self.DQN_id:
                self.action = (source,destination)



    # old code
    #
    # def gameover(self):
    #     players = range(1, len(self.players))
    #     living = list(filter(self.is_alive, players))
    #     if len(living) == 1:
    #         return living[0], count_ships(self.planets, self.fleets)
    #     elif self.turn >= 200:
    #         ship_counts = count_ships(self.planets, self.fleets)
    #         ship_counts = [(p, s) for p, s in ship_counts if p > 0]
    #         winner = 0 if ship_counts[0][1] == ship_counts[1][1] else ship_counts[0][0]
    #         return winner, ship_counts
    #     else:
    #         return -1, []

    # def is_alive(self, player):
    #     for planet in self.planets:
    #         if planet.owner == player:
    #             return True
    #     return False

# Endgame Conditions http://planetwars.aichallenge.org/specification.php
#
# The following conditions will cause the game to end:
#
# - The turn limit is reached. The winner is the player with the most ships,
#   both on planets and in fleets. If both# players have the same number of
#   ships, it's a draw.
#
# - One player runs out of ships and planets entirely. The winner is the other
#   player.
#
# - Both players run out of ships and planets at the same time. The game is a
#   draw.
#
# - A bot sends invalid data and forfeits the game.
#
# - A bot crashes and forfeits the game.
#
# - A bot exceeds the time limit without completing its orders (it never sends
#   a line that says 'go') it forfeits the game.
#
# - A bot attempts to do something that the tournament manager deems a
#   security issue and is disqualified.

    # returns winning player, list of (player, ship_count) for live players, and #turns
    # return (0,*,turns) if draw
    def gameover(self):
        players = range(1, len(self.players))
        # living players != neutral
        living = list(filter(self.is_alive, players))
        if len(living) == 0:
          return 0, [], self.turn # all dead: draw
        if len(living) == 1:
          # count_ships returns descending (p,count) pairs
          return living[0], count_ships(self.planets, self.fleets), self.turn
        elif self.turn >= 200:
          ship_counts = count_ships(self.planets, self.fleets)
          ship_counts = [(p, s) for p, s in ship_counts if p > 0]
          winner = 0 if ship_counts[0][1] == ship_counts[1][1] else ship_counts[0][0]
          return winner, ship_counts, self.turn
        else:
          return -1, [], self.turn

    def is_alive(self, player):

      for planet in self.planets:
        if planet.owner == player:
          return True
        
      for fleet in self.fleets:
        if fleet.owner == player:
          return True
              
      return False

    def save_weights(self):
        self.DQN_player.save_weights()

    def load_weights(self):
        self.DQN_player.load_weights()

    def counter_epoch(self):
        return self.DQN_player.counter
                
