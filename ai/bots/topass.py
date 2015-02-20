# -*- coding: utf-8 -*-

import sys
import math
import numpy
import random
import itertools

from .. import planetwars_class
from planetwars.datatypes import Order
from planetwars.utils import *

# TODO: Evaluate the performance of the NN on other bots outside of the training, separately?
# TODO: Shuffle the input/output vectors to force abstracting across planet order.

# DONE: Scale the outputs to 9 (weak|closest|best union friendly|enemy|neutral).
# DONE: Add StrongToClose and StrongToWeak behavior as an epsilon percentage.
# DONE: Experiment with incrementally decreasing learning rate, or watch oscillations in performance.
# DONE: Evaluate against known bots at regular intervals, to measure performance.

from ..bots.nn.deepq.deepq import DeepQ
from .stochastic import Stochastic
from .sample import strong_to_weak, strong_to_close


ACTIONS = 9
SCALE = 1.0

def split(index, stride):
    return index / stride, index % stride


@planetwars_class
class DeepNaN(object):

    def __init__(self):
        self.learning_rate = 0.0001
        self.bot = DeepQ([ # ("RectifiedLinear", 3500),
                           # ("RectifiedLinear", 3500),
                          #("RectifiedLinear", 1500),
                           # ("RectifiedLinear", 1500),
                           # ("RectifiedLinear", 2000),
                          ("RectifiedLinear", 2000),
                          ("Linear", )],
                         dropout=False, learning_rate=self.learning_rate)

        # try:
        #     self.bot.load()
        #     print "DeepNaN loaded!"
        # except IOError:
        #     pass

        self.turn_score = {}
        self.iteration_score = {}

        self.games = 0
        self.winloss = 0
        self.total_score = 0.0
        self.epsilon = 0.20001
        self.greedy = None
        self.iterations = 0
        self.previous = -1

    def __call__(self, turn, pid, planets, fleets):
        if pid == 1:
            # if self.games & 1 == 0:
            #    self.bot.epsilon = 0.1
            #    n_best = int(40.0 * self.epsilon)
            #if self.games & 1 != 0:
            self.bot.epsilon = self.epsilon
        else:
            self.bot.epsilon = 1.0

            # Hard-coded opponent bots use greedy policy (1-2*epsilon) of the time.
            if random.random() > self.epsilon * 2:
                if self.games & 2 == 0:
                    self.greedy = strong_to_close
                if self.games & 2 != 0:
                    self.greedy = strong_to_weak
                self.bot.epsilon = 0.0             

            # One of the three opponents is a fully randomized bot.
            # if self.games % 3 == 2:
            #    self.bot.epsilon = 1.0

        # Build the input matrix for data to feed into the DNN.
        a_inputs = self.createInputVector(pid, planets, fleets)
        orders, a_filter = self.createOutputVectors(pid, planets, fleets)

        # Reward calculated from the action in previous timestep.
        # score = sum([p.growth for p in planets if p.id == pid])
        # reward = (score - self.turn_score.get(pid, 0)) / 20.0
        # self.turn_score[pid] = score
        reward = 0.0

        order_id = self.bot.act_qs(a_inputs, reward * SCALE, episode=pid, terminal=False,
                                   q_filter=a_filter, n_actions=len(orders))

        if order_id is None or a_filter[order_id] <= 0.0:
            return []
        
        o = orders[order_id]
        assert o is not None, "The order specified is invalid."
        return o

    def done(self, turns, pid, planets, fleets, won):
        a_inputs = self.createInputVector(pid, planets, fleets)
        n_actions = len(planets) * ACTIONS + 1
        score = (+1.0 - float(turns)/301.5) if won else (-1.0 + float(turns)/301.5)

        self.bot.act_qs(a_inputs, int(won) * 2.0 - 1, terminal=True, n_actions=n_actions,
                        q_filter=0.0, episode=pid)

        if pid == 2:
            return
        
        n_best = int(20.0 * self.epsilon)
        self.games += 1
        self.total_score += score
        self.winloss += int(won) * 2 - 1

        score =  int(won) * 2 - 1
        # print '#', int(self.games), "(%i)" % len(self.bot.memory), self.total_score/self.games*2
        BATCH = 1
        if self.games % BATCH == 0:

            self.iterations += 1
            n_batch = 5000
            print "\nIteration %i with ratio %+i as score %f." % (self.iterations, self.winloss, self.total_score / BATCH)
            print "  - memory %i, latest %i, batch %i" % (len(self.bot.memory), len(self.bot.memory)-self.bot.last_training, n_batch)

            # if self.total_score < self.iteration_score.get(pid, -1.0):
            #    self.learning_rate /= 2.0
            #    self.bot.network.trainer.learning_rate.set_value(self.learning_rate)
            #    print "  - adjusting learning rate to %f" % (self.learning_rate,)
            # self.iteration_score[pid] = self.total_score
            #if(score !=self.previous):
            self.bot.train_qs(n_epochs=4, n_batch=n_batch)
            #else:
            #print "ignoring"
            """
            if len(self.bot.memory) > 1000000:                
                self.bot.network.epsilon = 0.000000002
                self.bot.train_qs(n_epochs=15, n_ratio=0.0, n_batch=n_batch)
            elif len(self.bot.memory) > 500000:
                self.bot.network.epsilon = 0.00000001
                self.bot.train_qs(n_epochs=5, n_ratio=0.0, n_batch=n_batch)
            elif len(self.bot.memory) > 250000:
                self.bot.network.epsilon = 0.00000004
                self.bot.train_qs(n_epochs=3, n_ratio=0.0, n_batch=n_batch)
            elif len(self.bot.memory) > 100000:
                self.bot.network.epsilon = 0.0000002
                self.bot.train_qs(n_epochs=1, n_ratio=0.0, n_batch=n_batch)
            """
            #self.bot.memory = self.bot.memory[len(self.bot.memory)/10:]

            # TODO: Measure impact of decaying the learning vs. learning speed.
            # if self.epsilon > 0.11:
            #   self.epsilon -= 0.05
            print "  - skills: top %i moves, or random %3.1f%%" % (n_best, self.epsilon * 100.0)
            #self.winloss = 0
            #self.total_score = 0.0
            if pid in self.turn_score:
                del self.turn_score[pid]

            #self.bot.save()

        self.previous = score

        if(self.games %100 == 0):
            self.winloss = 0
            self.total_score = 0
            #self.iteration = 0
        else:
            if turns >= 201:
                if won:
                    sys.stdout.write('o')
                else:
                    sys.stdout.write('.')
            else:
                if won:
                    sys.stdout.write('O')
                else:
                    sys.stdout.write('_')



    def createOutputVectors(self, pid, planets, fleets):        
        indices = range(len(planets))
        # random.shuffle(indices)

        if pid == 2:
            for i in range(1, len(planets), 2):
                indices[i], indices[i+1] = indices[i+1], indices[i]
                assert planets[i].growth == planets[i+1].growth

        # Global data used to create/filter possible actions.
        my_planets, their_planets, neutral_planets = aggro_partition(pid, planets)        
        other_planets = their_planets + neutral_planets

        order = None
        if self.greedy:
            g = self.greedy(0, pid, planets, fleets)
            if len(g) > 0:
                order = g[0]
            self.greedy = None

        action_valid = [
            bool(neutral_planets),   # WEAKEST NEUTRAL
            bool(neutral_planets),   # CLOSEST NEUTRAL
            bool(neutral_planets),   # BEST NEUTRAL
            bool(their_planets),     # WEAKEST ENEMY
            bool(their_planets),     # CLOSEST ENEMY
            bool(their_planets),     # BEST ENEMY
            len(my_planets) >= 2,    # WEAKEST FRIENDLY
            len(my_planets) >= 2,    # CLOSEST FRIENDLY
            len(my_planets) >= 2,    # BEST FRIENDLY
        ]

        # Matrix used as a multiplier (filter) for inactive actions.
        n_actions = len(planets) * ACTIONS + 1
        orders = []

        a_filter = numpy.zeros((n_actions,))
        for order_id in range(n_actions-1):
            src_id, act_id = split(order_id, ACTIONS)
            src_id = indices[src_id]
            src = planets[src_id]
            if not action_valid[act_id] or src.owner != pid:
                orders.append([])
                continue

            if act_id == 0: # WEAKEST NEUTRAL
                dst = min(neutral_planets, key=lambda x: x.ships * 100 + turn_dist(src, x))
            if act_id == 1: # CLOSEST NEUTRAL
                dst = min(neutral_planets, key=lambda x: turn_dist(src, x) * 1000 + x.ships)
            if act_id == 2: # BEST NEUTRAL
                dst = max(neutral_planets, key=lambda x: x.growth * 10 - x.ships)
            if act_id == 3: # WEAKEST ENEMY
                dst = min(their_planets, key=lambda x: x.ships * 100 + turn_dist(src, x))
            if act_id == 4: # CLOSEST ENEMY
                dst = min(their_planets, key=lambda x: turn_dist(src, x) * 1000 + x.ships)
            if act_id == 5: # BEST ENEMY
                dst = max(their_planets, key=lambda x: x.growth * 10 - x.ships)
            if act_id == 6: # WEAKEST FRIENDLY
                dst = min(set(my_planets)-set(src), key=lambda x: x.ships * 100 + turn_dist(src, x))
            if act_id == 7: # CLOSEST FRIENDLY
                dst = min(set(my_planets)-set(src), key=lambda x: turn_dist(src, x) * 1000.0 + x.ships)
            if act_id == 8: # BEST FRIENDLY
                dst = max(set(my_planets)-set(src), key=lambda x: x.growth * 10 - x.ships)

            if dst.id == src.id:
                orders.append(None)
                continue

            if order is None:
                orders.append([Order(src, dst, src.ships * 0.5)])
                a_filter[order_id] = 1.0
            else:
                if order.source.id == src_id and order.destination.id == dst.id:
                    orders.append([Order(src, dst, src.ships * 0.5)])
                    a_filter[order_id] = 1.0
                    order = None
                else:
                    orders.append(None)

        # NO-OP.
        a_filter[-1] = 1.0
        orders.append([])

        assert order is None, "Neural network did not support the greedy order suggested."
        return orders, a_filter


    def createInputVector(self, pid, planets, fleets):
        indices = range(len(planets))
        # random.shuffle(indices)

        if pid == 2:
            for i in range(1, len(planets), 2):
                indices[i], indices[i+1] = indices[i+1], indices[i]
                assert planets[i].growth == planets[i+1].growth

        # 1) Three layers of ship counters for each faction.
        a_ships = numpy.zeros((len(planets), 3))
        for p in planets:
            if p.owner == 0:
               a_ships[indices[p.id], 0] = 1+p.ships
            if p.owner == pid:
               a_ships[indices[p.id], 1] = 1+p.ships
            if p.owner != pid:
               a_ships[indices[p.id], 2] = 1+p.ships               

        # 2) Growth rate for all planets.
        a_growths = numpy.array([planets[i].growth for i in indices])

        # 3) Distance matrix for planet pairs.
        a_dists = numpy.zeros((len(planets), len(planets)))
        for A, B in itertools.product(planets, planets):
            a_dists[indices[A.id], indices[B.id]] = dist(A, B)

        # 4) Incoming ships bucketed by arrival time (logarithmic)
        n_buckets = 12
        a_buckets = numpy.zeros((len(planets), n_buckets))
        for f in fleets:
            d = math.log(f.remaining_turns) * 4
            a_buckets[indices[f.destination], min(n_buckets-1, d)] += f.ships * (1 if f.owner == pid else -1)

        # Full input matrix that combines each feature.
        a_inputs = numpy.concatenate((a_ships.flatten(), a_growths, a_dists.flatten(), a_buckets.flatten()))
        return a_inputs.astype(numpy.float32) / 1000.0
