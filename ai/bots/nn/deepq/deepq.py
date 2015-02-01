__author__ = 'ssamot, schaul'

from ai.bots.nn.sknn.sknn import sknn, IncrementalMinMaxScaler
import collections
import numpy as np
import cPickle as pickle



class DeepQ(object):
    """
    A Q learning agent
    """

    def __init__(self, layers,  dropout=False, input_scaler=None, output_scaler=None, learning_rate=0.001, verbose=0):
        self.max_memory = 500000
        self.memory = []
        self.network = sknn(layers, dropout, input_scaler, output_scaler, learning_rate,verbose)
        ##self.target_network = pylearn2MLPO()
        self.target_network = self.network
        self.gamma = 0.95
        self.epsilon = 0.15
        print 'gamma', self.gamma, 'epsilon', self.epsilon, 'lr', learning_rate

        self.initialised = False

        self.memory = []
        self.episodes = collections.defaultdict(list)
        self.last_sa = None
        self.last = {}

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

          self.network.fit(inputs, targets, 10)
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
            sa = state.reshape(1, state.size)
            target = np.zeros((1, n_actions))
            self.network.linit(sa, target)
            self.initialised = True

        # Compute the next Q-values for all actions in this state.  These are 
        # filtered based on which are available, specified by game logic.
        qs = self.__Qs(np.array([state]))[0] * q_filter
        if np.random.random() < self.epsilon:
            action = np.random.randint(0, n_actions)
        else:
            action = qs.argmax()

        last_s, last_q = self.last.get(episode, (None, None))
        if last_s is not None and not terminal:
            self.episodes[episode].append([last_s, last_q, action, reward])

        if terminal:            
            r, i = reward / self.gamma, 0           
            for ps, pq, action, reward in reversed(self.episodes[episode]):
                r = max(-1.0, min(+1.0, r * self.gamma + reward))
                # sqa.max() * (1.0 - self.epsilon) + self.epsilon * sqa.mean()
                # if reward > 0.0: 
                self.memory.append([ps, action, r])
                i += 1

            self.episodes[episode] = []
            self.last[episode] = (None, None)
        else:
            self.last[episode] = (state, qs)
        return action

    def train_qs(self, n_samples, n_epochs):
        updates = min(len(self.memory) / 2, n_samples)
        inputs = np.zeros((updates, self.memory[0][0].size))
        targets = np.zeros((updates, self.n_actions))

        current = self.network.predict(np.array([m[0] for m in self.memory]))

        i, threshold = 0, 0.5
        min_e, max_e = float("inf"), -float("inf")
        while i < updates:
            r = np.random.randint(len(self.memory))
            state, action, reward = self.memory[r]
            e = abs(current[r][action] - reward)
            if e < threshold:
                threshold -= 0.0000001
                continue

            min_e = min(e, min_e)
            max_e = max(e, max_e)

            mask = np.zeros((self.n_actions))
            mask[action] = 1.0
            targets[i] = current[r] * (1.0 - mask) + reward * mask
            inputs[i] = state
            i += 1
        print "  - samples %i: %f / %f" % (updates, min_e, max_e)

        self.network.fit(inputs, targets, epochs=n_epochs)
        predicted = self.network.predict(inputs)
        error = (targets - predicted) ** 2
        print "  - error %f / %f / %f" % (error.min(), error.mean(), error.max())
        print "  - predicted %f / %f / %f" % (predicted.min(), predicted.mean(), predicted.max())
        print "  - targets %f / %f / %f" % (targets.min(), targets.mean(), targets.max())

        memory = []
        for i, m in enumerate(self.memory):
            state, action, reward = m
            if abs(current[i][action] - reward) > 0.05:
                memory.append(m)
        print "  - pruned %i" % (len(self.memory) - len(memory))
        self.memory = memory

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
        pickle.dump(self, open(out_path, "wb"))


    @staticmethod
    def load():
        try:
            with open("./dq.pickle","r") as f:
                return pickle.load(f)
        except IOError as e:
            raise RuntimeError(e)


    def __getstate__(self):
        #print self.__dict__.keys()
        tbr =  dict((k, v) for (k, v) in self.__dict__.iteritems() if k != "memory")
        tbr["memory"] = []
        return tbr

