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


ACTIONS = 4


@planetwars_class
class TopAss(object):

    def __init__(self):
        self.bot = DeepQ([("RectifiedLinear", 1000),
                          ("RectifiedLinear", 1000),
                          ("RectifiedLinear", 1000),
                          ("Linear", )],
                         dropout=True, learning_rate=0.0001)

        try:
            self.bot.load()
            print "TopAss loaded!"
        except IOError:
            pass

        self.last_score = {}
        self.games = 0
        self.winloss = 0
        self.total_reward = 0.0
        self.epsilon_enemy = 0.05
        self.epsilon_friend = 0.5

    def __call__(self, turn, pid, planets, fleets):
        if pid == 1:
            # First player plays randomly, from 85% to 5%.
            self.bot.epsilon = self.epsilon_friend
        else:
            # Second player plays greedily, from 5% to 85%.
            self.bot.epsilon = 1.0

        # Build the input matrix for data to feed into the DNN.
        a_inputs = self.createInputVector(pid, planets, fleets)

        # Global data used to create/filter possible actions.
        my_planets, their_planets, neutral_planets = aggro_partition(pid, planets)        
        other_planets = their_planets + neutral_planets

        action_valid = {
            0: bool(neutral_planets),   # WEAKEST NEUTRAL
            1: bool(their_planets),     # WEAKEST ENEMY
            2: bool(neutral_planets),   # CLOSEST NEUTRAL
            3: bool(their_planets)      # CLOSEST ENEMY
        }

        # Matrix used as a multiplier (filter) for inactive actions.
        n_actions = len(planets) * ACTIONS
        qf = numpy.zeros((n_actions,))
        for order_id in range(n_actions):
            src_id, act_id = order_id / ACTIONS, order_id % ACTIONS
            if not action_valid[act_id]:
                continue
            if planets[src_id].owner != pid:
                continue
            qf[order_id] = 1.0

        # Reward calculated from the action in previous timestep.
        score = sum([p.growth for p in planets if p.id == pid])
        reward = (score - self.last_score.get(pid, 0)) / 20.0
        if reward < 0: reward /= 4.0
        self.last_score[pid] = score
        order_id = self.bot.act_qs(a_inputs, reward,
                                   terminal=False, n_actions=n_actions, q_filter=qf, episode=pid)

        if order_id is None:
            return []

        src_id, act_id = order_id / ACTIONS, order_id % ACTIONS
        src = planets[src_id]
        order_f = qf[order_id]

        if order_f <= 0.0:
            return []

        assert src.owner == pid, "The order (%i -> %i) is invalid == %f." % (src_id, dst_id, order_f)
        assert action_valid[act_id], "This action type is currently invalid."

        if act_id == 0 and neutral_planets:
            # print "WEAKEST NEUTRAL", src_id
            dst = min(neutral_planets, key=lambda x: x.ships)
            return [Order(src, dst, src.ships * 0.5)]
        if act_id == 1 and their_planets:
            # print "WEAKEST ENEMY", src_id
            dst = min(their_planets, key=lambda x: x.ships)
            return [Order(src, dst, src.ships * 0.5)]
        if act_id == 2 and neutral_planets:
            # print "CLOSEST NEUTRAL", src_id
            dst = min(neutral_planets, key=lambda x: turn_dist(src, x))
            return [Order(src, dst, src.ships * 0.5)]
        if act_id == 3 and their_planets:
            # print "CLOSEST ENEMY", src_id
            dst = min(their_planets, key=lambda x: turn_dist(src, x))
            return [Order(src, dst, src.ships * 0.5)]

        return []

    def done(self, turns, pid, planets, fleets, won):
        a_inputs = self.createInputVector(pid, planets, fleets)
        n_actions = len(planets) * ACTIONS
        if turns == 201:
            score = +0.5 if won else -0.25
        else:
            score = +1.0 if won else -0.5

        self.bot.act_qs(a_inputs, score, terminal=True, n_actions=n_actions,
                        q_filter=0.0, episode=pid)

        assert pid == 1

        self.games += 1
        self.total_reward += score
        self.winloss += int(won) * 2 - 1

        # print '#', int(self.games), "(%i)" % len(self.bot.memory), self.total_reward/self.games*2
        BATCH = 100
        if self.games % BATCH == 0:
            n_batch = 5000
            print "\nIteration %i with ratio %+i as score %f." % (self.games/BATCH, self.winloss, self.total_reward / BATCH)
            print "  - memory %i, latest %i, batch %i" % (len(self.bot.memory), len(self.bot.memory)-self.bot.last_training, n_batch)
            
            self.bot.train_qs(n_epochs=2, n_ratio=0.5, n_batch=n_batch)
            self.bot.memory = self.bot.memory[len(self.bot.memory)/100:]

            # if self.winloss > -BATCH / 3:
            #    if self.epsilon_enemy < 0.85:
            #        self.epsilon_enemy += 0.01
            if self.epsilon_friend > 0.1:
                self.epsilon_friend -= 0.02
            print "  - skills: random %i%% (self) vs. greedy %i%% (other)" % (self.epsilon_friend * 100.0, self.epsilon_enemy * 100.0)
            self.winloss = 0
            self.total_reward = 0.0
            del self.last_score[pid]

            self.bot.save()
        else:
            if turns == 201:
                if score > 0.0:
                    sys.stdout.write('·')
                elif score < 0.0:
                    sys.stdout.write('-')
            else:
                if score > 0.0:
                    sys.stdout.write('+')
                elif score < 0.0:
                    sys.stdout.write('=')

    def createInputVector(self, pid, planets, fleets):
        indices = range(len(planets))
        # random.shuffle(indices)
        assert pid != 2
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
        a_inputs = numpy.concatenate((a_ships.flatten(), a_buckets.flatten()))
        # a_growths
        # a_dists.flatten(), 
        return a_inputs.astype(numpy.float32) / 1000.0
