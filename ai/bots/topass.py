# -*- coding: utf-8 -*-

import sys
import math
import numpy
import random
import itertools

from .. import planetwars_class
from planetwars.datatypes import Order
from planetwars.utils import *

from ..bots.nn.deepq.deepq import DeepQ
from .stochastic import Stochastic


@planetwars_class
class TopAss(object):

    def __init__(self):
        self.bot = DeepQ([("RectifiedLinear", 1500), ("RectifiedLinear", 1500), ("RectifiedLinear", 1500), ("Linear", )],
                         dropout=True, learning_rate=0.0005)

        self.last_score = None
        self.games = 0
        self.winloss = 0
        self.total_reward = 0.0
        self.epsilon = 1.0

    def __call__(self, turn, pid, planets, fleets):
        if pid == 1:
            self.bot.epsilon = 0.15
        else:
            self.bot.epsilon = self.epsilon

        a_inputs = self.createInputVector(pid, planets, fleets)

        score = sum([p.growth for p in planets if p.id == pid])
        reward = score - self.last_score if self.last_score else 0.0        
        self.last_score = score

        action_count = len(planets) * len(planets)
        qf = numpy.zeros((action_count,))
        for i in range(action_count):
            src_id, dst_id = i % len(planets), i / len(planets)
            if planets[src_id].owner == pid:
                qf[i] = 1.0

        order_id = self.bot.act_qs(a_inputs, 0.25 if reward > 0.0 else (-0.25 if reward < 0.0 else 0.0),
                                   terminal=False, n_actions=action_count, q_filter=qf, episode=pid)

        src_id, dst_id = order_id % len(planets), order_id / len(planets)
        src, dst = planets[src_id], planets[dst_id]
        order_f = qf[order_id]       

        if src_id != dst_id and order_f > 0.0:
            assert src.owner == pid, "The order (%i -> %i) is invalid == %f." % (src_id, dst_id, order_f)
            return [Order(src, dst, src.ships * 0.5)]
        else:
            return []

    def done(self, turns, pid, planets, fleets, won):
        a_inputs = self.createInputVector(pid, planets, fleets)
        n_actions = len(planets) * len(planets)
        if turns == 200:
            score = +0.1 if won else -0.1
        else:
            score = +1.0 if won else -1.0

        self.bot.act_qs(a_inputs, score, terminal=True, n_actions=n_actions,
                        q_filter=0.0, episode=pid)

        if pid != 1:
            return

        self.games += 1
        self.total_reward += score
        self.winloss += int(score)

        # print '#', int(self.games), "(%i)" % len(self.bot.memory), self.total_reward/self.games*2
        BATCH = 100
        if self.games % BATCH == 0:
            print "\nIteration %i with ratio %+i as score %f." % (self.games/BATCH, self.winloss, self.total_reward / self.games)
            print "  - memory %i" % (len(self.bot.memory))
            
            self.bot.train_qs(n_samples=50000, n_epochs=10)

            if self.winloss > -BATCH / 3 and self.epsilon > 0.15:
                self.epsilon -= 0.05
                print "  - skill now %f" % (1 - self.epsilon)
            self.winloss = 0
        else:
            if turns == 200:
                sys.stdout.write('▪')
            elif score > 0.0:
                sys.stdout.write('+')
            elif score < 0.0:
                sys.stdout.write('-')

    def createInputVector(self, pid, planets, fleets):
        # 1) Three layers of ship counters for each faction.
        a_ships = numpy.zeros((len(planets), 3))
        for p in planets:
            if p.owner == 0:
               a_ships[p.id, 0] = p.ships
            if p.owner == pid:
               a_ships[p.id, 1] = p.ships
            if p.owner != pid:
               a_ships[p.id, 2] = p.ships

            # 1) a_ships[p.id] = p.ships if p.owner == pid else -p.ships

            """
            2)
            a_ships[p.id] = p.ships
            owner = -1.0
            if p.id == pid: owner = +1.0
            if p.id == 0: owner = 0.0
            a_owners[p.id] = owner
            """

        # 2) Growth rate for all planets.
        a_growths = numpy.array([p.growth for p in planets])

        # 3) Distance matrix for planet pairs.
        a_dists = numpy.zeros((len(planets), len(planets)))
        for A, B in itertools.product(planets, planets):
            if A.id != B.id:
                a_dists[A.id, B.id] = dist(A, B)

        # 4) Incoming ships bucketed by arrival time (logarithmic)
        n_buckets = 12
        a_buckets = numpy.zeros((len(planets), n_buckets))
        for f in fleets:
            d = math.log(f.remaining_turns) * 4
            a_buckets[f.destination, min(n_buckets-1, d)] += f.ships * (1 if f.owner == pid else -1)

        # Full input matrix that combines each feature.
        a_inputs = numpy.concatenate((a_ships.flatten(), a_growths, a_dists.flatten(), a_buckets.flatten()))
        return a_inputs / 1000.0
