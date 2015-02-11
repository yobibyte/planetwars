# -*- coding: utf-8 -*-

__author__ = 'ssamot, schaul'

import sys
import math
import random
import collections
import cPickle as pickle

from ai.bots.nn.sknn.sknn import sknn, IncrementalMinMaxScaler
import numpy as np


class DeepQ(object):
    """
    A Q learning agent
    """

    def __init__(self, layers, dropout=False, input_scaler=None, output_scaler=None, learning_rate=0.001, verbose=0):
        self.network = sknn(layers, dropout, input_scaler, output_scaler, learning_rate, verbose)
        
        self.gamma = 0.99
        self.epsilon = 0.15
        self.n_best = 1

        self.initialised = False

        self.memory = []
        self.last_training = 0
        
        # Storage for information about multiple episodes at the same time, e.g for
        # adversarial search.
        self.episodes = collections.defaultdict(list)
        self.last = {}

        # Save numpy arrays so resizng/reallocating doesn't cause fragmentation on GPU.
        self.inputs = None
        self.targets = None

        self.last_sa = None

    def addToMemory(self, *args):
        self.memory += [args]

    def train_from_memory(self, updates):

        if len(self.memory) > 1000:
          updates = min(len(self.memory)/2, updates)
          inputs = np.zeros((updates, self.memory[0][0].size))
          targets = np.zeros((updates, 1))
          for update in range(updates):

            r = np.random.randint(len(self.memory))
            input = self.memory[r]
            target = self.computeTarget(*input)

            inputs[update] = input[0]
            targets[update] = target

          self.network.fit(inputs, targets, 20)
        else:
           print "not enough"
        if len(self.memory) > self.max_memory:
           print "clipped memory"
           self.memory = self.memory[self.max_memory/2:]


    def __Qs(self,sas):
        Q = self.network.predict(sas)
        return Q

    def computeTarget(self, last_sa, reward, terminal, all_next_sas):
        gamma = self.gamma
        V = 0
        if terminal == 0:
            Qs = self.__Qs(all_next_sas)
            maxQ = Qs.max()
            V = maxQ * (1-self.epsilon) + self.epsilon * Qs.mean()
            #print maxQ

        target = reward  + (1-terminal) * gamma * V

        return target

    def fit(self,last_sa, reward, terminal, all_next_sas):
        gamma = self.gamma
        maxQ = 0
        if terminal == 0:
            maxQ = self.__Qs(all_next_sas).max()

        target = reward  + (1-terminal) * gamma * maxQ
        self.network.fit(last_sa.reshape(1,last_sa.size), np.array([[target]]))
        if self.swap_counter % self.swap_iterations == 0:
            pass

    def act_qs(self, state, reward, terminal, n_actions, q_filter, episode=0):
        # Make sure the deep neural network has been correctly initialized given
        # the exact input and output dimensions.
        self.n_actions = n_actions
        if not self.initialised:
            print 'S->A', state.shape, n_actions
            sa = state.reshape(1, state.size)
            target = np.zeros((1, n_actions))
            self.network.linit(sa, target)
            self.initialised = True

        if not terminal:
            # Compute the next Q-values for all actions in this state.  These are 
            # filtered based on which are available, specified by game logic.
            qs = self.__Qs(np.array([state]))[0] - 100.0 * (1.0 - q_filter)
            # Determine the argmax for multiple possible options.
            indices = np.argpartition(qs, -self.n_best)[-self.n_best:]

            non_zero = q_filter.nonzero()[0]
            valid_actions = list(set(non_zero)-set(indices))

            if np.random.random() < self.epsilon:
                if len(valid_actions):
                    action = random.choice(valid_actions)
                    probability = (self.epsilon, len(valid_actions), q_filter)
                else:
                    action = None
                    probability = None
            else:                        
                action = np.random.choice(indices)
                probability = ((1.0 - self.epsilon), self.n_best, q_filter)
        else:
            action = None
            probability = None

        last_s = self.last.get(episode, None)
        if last_s is not None and not terminal:
            self.episodes[episode].append([last_s, action, probability, reward])

        if terminal:
            r, i = reward / self.gamma, 0           
            for ps, action, probability, reward in reversed(self.episodes[episode]):
                r = max(-1.0, min(+1.0, r * self.gamma + reward))
                # sqa.max() * (1.0 - self.epsilon) + self.epsilon * sqa.mean()
                # if reward > 0.0: 
                if action is not None:
                    self.memory.append([ps, action, probability, r])
                i += 1

            self.episodes[episode] = []
            self.last[episode] = None
        else:
            self.last[episode] = state
        return action

    def train_qs(self, n_epochs, n_batch=5000):        
        if self.inputs is None:
            self.inputs = np.zeros((n_batch, self.memory[0][0].size), dtype=np.float32)
        if self.targets is None:
            self.targets = np.zeros((n_batch, self.n_actions), dtype=np.float32)

        error_stats = [+float("inf"), 0.0, -float("inf")]
        target_stats = [+float("inf"), 0.0, -float("inf")]
        pred_stats = [+float("inf"), 0.0, -float("inf")]

        n_epochs = int(math.ceil(n_epochs * len(self.memory) / n_batch))
        print "  - training %i epochs of batch size %i" % (n_epochs, n_batch)
        print "  - ",
        prune = set()
        for e in range(n_epochs):
            batch = []
            for i in range(n_batch):
                j = np.random.randint(0, len(self.memory))
                state, action, probability, reward = self.memory[j]
                self.inputs[i] = state
                batch.append((action, probability, reward, j))
        
            original = self.network.predict(self.inputs)
            for i, (action, probability, reward, _) in enumerate(batch):
                mask = np.zeros((self.n_actions), dtype=np.float32)
                mask[action] = 1.0

                # Extract original information, and calculate probability.
                prob, count, q_filter = probability
                o = prob / count

                qs = original[i] * -100.0 * (1.0 - q_filter)
                # Recalculate the current probability given current policy.
                indices = np.argpartition(qs, -self.n_best)[-self.n_best:]
                if action in indices:
                    p = (1.0 - self.epsilon) / self.n_best
                    # print action, indices
                    # assert p == o, "argmax: %f == %f" % (p, o)
                else:
                    p = self.epsilon / count
                    # print action, indices
                    # assert p == o, "random: %f == %f" % (p, o)

                # Importance sampling, knowing the old probability o and the current p,
                # renormalize the reward to closer to what it should be now.                
                r = max(-1.0, min(+1.0, reward * (o / p)))

                self.targets[i] = original[i] * (1.0 - mask) + r * mask

            self.network.fit(self.inputs, self.targets, epochs=1)
            predicted = self.network.predict(self.inputs)

            error = []
            for i, (action, probability, reward, _) in enumerate(batch):
                e = (predicted[i][action] - reward) ** 2.0
                error.append(e)
            error = np.array(error)
            error_stats[0] = min(error_stats[0], error.min())
            error_stats[1] += error.mean() / float(n_epochs)
            error_stats[2] = max(error_stats[2], error.max())

            target_stats[0] = min(target_stats[0], self.targets.min())
            target_stats[1] += self.targets.mean() / float(n_epochs)
            target_stats[2] = max(target_stats[2], self.targets.max())

            pred_stats[0] = min(pred_stats[0], predicted.min())
            pred_stats[1] += predicted.mean() / float(n_epochs)
            pred_stats[2] = max(pred_stats[2], predicted.max())

            threshold = 0.0 # TODO: Measure impact of discarding easy to approximate samples.
            for i, (action, probability, reward, index) in enumerate(batch):
                if (predicted[i][action] - reward) ** 2 <= threshold:
                    prune.add(index)

            sys.stdout.write('■'); sys.stdout.flush()

        print "\r" + " " * (n_epochs + 4),
        print "\r  - target %f / %f / %f" % tuple(target_stats)
        print "  - pred %f / %f / %f" % tuple(pred_stats)
        print "  - error %f / %f / %f" % tuple(error_stats)
        if len(prune):
            print "  - pruned %i with threshold %f" % (len(prune), threshold)
        self.memory = [m for i, m in enumerate(self.memory) if i not in prune]        
        self.last_training = len(self.memory)

    # e-greedy
    def act(self,all_next_sas, reward, terminal):

        if(not self.initialised):
            sa = all_next_sas[0].reshape(1,all_next_sas[0].size)
            target = np.array([[0]])
            #print target.shape, sa.shape
            self.network.fit(sa,target)
            self.initialised = True

        if(np.random.random() < self.epsilon):
            r =  np.random.randint( 0,len(all_next_sas))
            #print "returning random action", r
            action =  r
        else:
            #print "returning best action", b_action
            Q = self.__Qs(all_next_sas)
            maxQ = Q.max()
            actions = []
            for i,q in enumerate(Q):
                if(q == maxQ):
                    actions.append(i)
            #print actions
            b_action = actions[np.random.randint(len(actions))]

            #print b_action
            action =  b_action

        last_sa = self.last_sa

        if(last_sa is not None):
            self.addToMemory(last_sa, reward, terminal, all_next_sas)


        self.last_sa = all_next_sas[action]

        return action

    def save(self):
        out_path = "./dq.pickle"
        pickle.dump(self.network.mlp, open(out_path, "wb"))

    def load(self):
        with open("./dq.pickle","r") as f:
            self.network.mlp = pickle.load(f)

    def __getstate__(self):
        #print self.__dict__.keys()
        tbr =  dict((k, v) for (k, v) in self.__dict__.iteritems() if k != "memory")
        tbr["memory"] = []
        return tbr

