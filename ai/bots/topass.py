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
from .sample import strong_to_weak, strong_to_close

@planetwars_class
class TopAss(object):

    def __init__(self):
        self.bot = DeepQ([("RectifiedLinear", 750), ("RectifiedLinear", 750),
                          ("RectifiedLinear", 750),
                          ("Linear", )],
                         dropout=True, learning_rate=0.0005)

        self.last_score = None
        self.games = 0
        self.winloss = 0
        self.total_reward = 0.0
        self.epsilon_enemy = 0.05
        self.epsilon_friend = 0.85

    def __call__(self, turn, pid, planets, fleets):
        if pid == 1:
            # First player plays randomly, from 85% to 5%.
            self.bot.epsilon = self.epsilon_friend
        else:
            # Second player plays greedily, from 5% to 85%.
            self.bot.epsilon = 1.0

        a_inputs = self.createInputVector(pid, planets, fleets)

        score = sum([p.growth for p in planets if p.id == pid])
        reward = score - self.last_score if self.last_score else 0.0        
        self.last_score = score

        action_count = len(planets) * len(planets)
        qf = numpy.zeros((action_count,))
        if numpy.random.random() >= 0.5:
            greedy = strong_to_weak(turn, pid, planets, fleets)
        else:
            greedy = strong_to_close(turn, pid, planets, fleets)

        if pid == 1 or not greedy or numpy.random.random() > self.epsilon_enemy:
            for i in range(action_count):
                src_id, dst_id = i % len(planets), i / len(planets)
                if src_id == dst_id: # Comment this to allow NOP.
                    continue
                if planets[src_id].owner == pid:
                    qf[i] = 1.0
            is_greedy = False
        else:
            is_greedy = True
            i = greedy[0].destination.id * len(planets) + greedy[0].source.id
            qf[i] = 1.0

        r = reward / 20.0
        order_id = self.bot.act_qs(a_inputs, +r if reward > 0.0 else (-r if reward < 0.0 else 0.0),
                                   terminal=False, n_actions=action_count, q_filter=qf, episode=pid)

        if order_id is None:
            return []

        src_id, dst_id = order_id % len(planets), order_id / len(planets)
        src, dst = planets[src_id], planets[dst_id]
        order_f = qf[order_id]

        if is_greedy:
            assert src_id == greedy[0].source.id
            assert dst_id == greedy[0].destination.id

        if order_f > 0.5:
            assert src_id != dst_id, "Not expecting source planet to equal the destination."
            assert src.owner == pid, "The order (%i -> %i) is invalid == %f." % (src_id, dst_id, order_f)
            return [Order(src, dst, src.ships * 0.5)]
        else:
            return []

    def done(self, turns, pid, planets, fleets, won):
        a_inputs = self.createInputVector(pid, planets, fleets)
        n_actions = len(planets) * len(planets)
        if turns == 201:
            score = +0.5 if won else -0.5
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
            n_samples = 20000
            print "\nIteration %i with ratio %+i as score %f." % (self.games/BATCH, self.winloss, self.total_reward / BATCH)
            print "  - memory %i with batch %i" % (len(self.bot.memory), n_samples)
            
            self.bot.train_qs(n_samples=n_samples, n_epochs=20)
            self.bot.memory = self.bot.memory[len(self.bot.memory)/100:]

            if self.winloss > -BATCH / 3:
                if self.epsilon_enemy < 0.85:
                    self.epsilon_enemy += 0.01
            if self.epsilon_friend > 0.05:
                self.epsilon_friend -= 0.01
            print "  - skills: random %i%% (self) vs. greedy %i%% (other)" % (self.epsilon_friend * 100.0, self.epsilon_enemy * 100.0)
            self.winloss = 0
            self.total_reward = 0.0

            # self.bot.save()
        else:
            if turns == 201:
                sys.stdout.write('·')
            elif score > 0.0:
                sys.stdout.write('+')
            elif score < 0.0:
                sys.stdout.write('-')

    def createInputVector(self, pid, planets, fleets):
        indices = range(len(planets))
        # random.shuffle(indices)
        if pid == 2:
            for i in range(1, len(planets), 2):
                indices[i], indices[i+1] = indices[i+1], indices[i]

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
