import sys
import numpy
import random
import itertools

from .. import planetwars_class
from planetwars.datatypes import Order
from planetwars.utils import *

from ..bots.nn.deepq.deepq import DeepQ


@planetwars_class
class TopAss(object):

    def __init__(self):
        self.bot = DeepQ([("RectifiedLinear", 500), ("RectifiedLinear", 500), ("Linear", )],
                         dropout=True, learning_rate=0.0001)

        self.last_score = None
        self.last_size = 0
        self.games = 0
        self.winloss = 0
        self.total_reward = 0.0

    def __call__(self, turn, pid, planets, fleets):
        # 1) Three layers of ship counters for each faction.
        a_ships = numpy.zeros((len(planets), 3))
        a_owners = numpy.zeros((len(planets)))
        for p in planets:
            a_ships[p.id, p.owner] = p.ships

            """
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

        # 4) Incoming ships bucketed by arrival time.
        n_buckets = 25
        a_buckets = numpy.zeros((len(planets), n_buckets))
        for f in fleets:
            a_buckets[f.destination, min(n_buckets-1, f.remaining_turns)] += f.ships * (1 if f.owner == pid else -1)

        # Full input matrix that combines each feature.
        a_inputs = numpy.concatenate((a_ships.flatten(), a_growths, a_dists.flatten())) # a_buckets.flatten()

        score = sum([p.growth for p in planets if p.id == pid])
        reward = score - self.last_score if self.last_score else 0.0        
        self.last_score = score

        action_count = len(planets) * len(planets)
        qf = numpy.zeros((action_count,))
        for i in range(action_count):
            src_id, dst_id = i % len(planets), i / len(planets)
            if planets[src_id].owner == pid:
                qf[i] = 1.0

        order_id = self.bot.act_qs(a_inputs, 1.0 if reward > 0.0 else (-1.0 if reward < 0.0 else 0.0),
                                   terminal=False, n_actions=action_count, q_filter=qf)

        src_id, dst_id = order_id % len(planets), order_id / len(planets)
        src, dst = planets[src_id], planets[dst_id]
        order_f = qf[order_id]       

        if src_id != dst_id and order_f > 0.0:
            assert src.owner == pid, "The order (%i -> %i) is invalid == %f." % (src_id, dst_id, order_f)
            return [Order(src, dst, src.ships * 0.5)]
        else:
            return []

    def done(self, won, turns):
        self.games += 1
        self.total_reward += float(won) - 0.5
        self.winloss += int(won) * 2 - 1

        # print '#', int(self.games), "(%i)" % len(self.bot.memory), self.total_reward/self.games*2
        BATCH = 100
        if self.games % BATCH == 0:
            print "\nIteration %i as %+i with ratio %f." % (self.games/BATCH, self.winloss, self.total_reward * 2.0 / self.games)
            print "  - memory %i" % (len(self.bot.memory))
            
            # self.bot.train_qs(n_updates=100000, n_last=len(self.bot.memory) - self.last_size)
            self.bot.train_qs(n_updates=100000, n_last=0, latest_ratio=0.0)
            self.winloss = 0
            self.bot.last_q = None
            self.bot.last_s = None
            self.bot.memory = self.bot.memory[len(self.bot.memory)/100:]
            self.last_size = len(self.bot.memory)
        else:
            sys.stdout.write('.')
