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
        self.learning_rate = 0.00001
        self.bot = DeepQ([("ConvRectifiedLinear", {"channels": 100, "kernel": (1,39)}),
                          # ("RectifiedLinear", 3500),
                          # ("RectifiedLinear", 3000),
                          # ("RectifiedLinear", 2500),
                          ("RectifiedLinear", 3000),
                          ("RectifiedLinear", 3000),
                          ("RectifiedLinear", 3000),
                          ("Linear", )],
                          dropout=False, learning_rate=self.learning_rate)

        try:
            self.bot.load()
            print "DeepNaN loaded!"
        except IOError:
            pass

        self.turn_score = {}
        self.iteration_score = {}

        self.games = 0
        self.winloss = 0
        self.total_score = 0.0
        self.epsilon = 0.20001
        self.greedy = None
        self.iterations = 0

    def __call__(self, turn, pid, planets, fleets):
        if pid == 2:
            if self.games % 2 == 0:
                return strong_to_weak(turn, pid, planets, fleets)

        # Build the input matrix for data to feed into the DNN.
        a_inputs = self.createInputVector(pid, planets, fleets)
        orders, a_filter = self.createOutputVectors(pid, planets, fleets)

        order_id = self.bot.act_qs(a_inputs, reward=0.0, episode=pid, terminal=False,
                                   q_filter=a_filter, n_actions=len(orders))

        if order_id is None or a_filter[order_id] <= 0.0:
            return []
        
        o = orders[order_id]
        assert o is not None, "The order specified is invalid."
        return o

    def done(self, turns, pid, planets, fleets, won):
        a_inputs = self.createInputVector(pid, planets, fleets)
        orders, a_filter = self.createOutputVectors(pid, planets, fleets)
        score = +1.0 if won else -1.0

        self.bot.act_qs(a_inputs, score * SCALE, terminal=True, n_actions=len(orders),
                        q_filter=0.0, episode=pid)

        if pid == 2:
            return
        
        if self.games % 2 == 0:            
            self.total_score += score
            self.winloss += (1 if won else -1) * (2 if turns < 201 else 1)
        self.games += 1

        n_batch = 400
        self.bot.train_qs(n_epochs=1, n_batch=n_batch)
        self.bot.memory = self.bot.memory[len(self.bot.memory)/250:]
        if pid in self.turn_score:
            del self.turn_score[pid]

        BATCH = 100
        if self.games % BATCH == 0:

            self.iterations += 1
            print "\nIteration %i with ratio %+i as score %f." % (self.iterations, self.winloss, self.total_score / BATCH)
            print "  - memory %i, latest %i, batch %i" % (len(self.bot.memory), len(self.bot.memory)-self.bot.last_training, n_batch)

            self.winloss = 0
            self.total_score = 0.0

            self.bot.save()
        else:
            if turns >= 201:
                sys.stdout.write('o' if won else '.')
            else:
                sys.stdout.write('O' if won else '_')

    """
    # SIMPLIFIED ACTION SPACE
    #
    def _createOutputVectors(self, pid, planets, fleets):        
        # Global data used to create/filter possible actions.
        my_planets, their_planets, neutral_planets = aggro_partition(pid, planets)        
        other_planets = their_planets + neutral_planets

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
            src_id = self.indices[src_id]
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

            orders.append([Order(src, dst, src.ships * 0.5)])
            a_filter[order_id] = 1.0

        # NO-OP.
        a_filter[-1] = 1.0
        orders.append([])

        del self.indices
        return orders, a_filter
    """

    def createOutputVectors(self, pid, planets, fleets):
        orders = []
        n_actions = len(planets) * len(planets)
        a_filter = numpy.zeros((n_actions,))

        for i, (src_id, dst_id) in enumerate(itertools.product(self.indices, repeat=2)):
            src = planets[src_id]
            if src.owner != pid:
                orders.append(None)
                continue

            a_filter[i] = 1.0
            if src_id == dst_id: # NOP
                orders.append([])
            else:
                orders.append([Order(src, planets[dst_id], src.ships * 0.5)])

        del self.indices
        return orders, a_filter

    def createInputVector(self, pid, planets, fleets):
        self.indices = range(len(planets))
        # random.shuffle(self.indices)

        if pid == 2:
            for i in range(1,len(planets),2):
                self.indices[i], self.indices[i+1] = self.indices[i+1], self.indices[i]
                assert planets[self.indices[i]].growth == planets[self.indices[i+1]].growth

        n_buckets = 12

        # For each planet:
        #   - Ship counts (3x)
        #   - Growth (1x)
        #   - Planet distances (N)
        #   - Incoming buckets (k)
        a_planets = numpy.zeros((len(planets), 3+1+len(planets)+n_buckets), dtype=numpy.float32)

        for p in planets:
            idx = self.indices[p.id]
            
            if p.owner == 0:   # NEUTRAL
               a_planets[idx, 0] = 1+p.ships
            if p.owner == pid: # FRIENDLY
               a_planets[idx, 1] = 1+p.ships
            if p.owner != pid: # ENEMY
               a_planets[idx, 2] = 1+p.ships

            # Ship creation per turn.
            a_planets[idx, 3] = p.growth

            # Distances from this planet.
            for o in planets:
                a_planets[idx, 4+self.indices[o.id]] = dist(p, o)

            # Incoming ships bucketed by arrival time (logarithmic)
            start = 4+len(planets)
            for f in [f for f in fleets if f.destination == p.id]:
                d = math.log(f.remaining_turns) * 4
                a_planets[idx, start+min(n_buckets-1, d)] += f.ships * (1.0 if f.owner == pid else -1.0)

        # Full input matrix that combines each feature.
        return a_planets / 1000.0
