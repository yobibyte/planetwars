__author__ = 'ssamot, schaul'

from ai.bots.nn.sknn.sknn import sknn, IncrementalMinMaxScaler
import numpy as np
import cPickle as pickle



class DeepQ(object):
    """
    A Q learning agent
    """

    def __init__(self, layers,  dropout=False, input_scaler=IncrementalMinMaxScaler(), output_scaler=IncrementalMinMaxScaler(), learning_rate=0.001, verbose=0):
        self.max_memory = 500000
        self.memory = []
        self.network = sknn(layers, dropout, input_scaler, None, learning_rate,verbose)
        ##self.target_network = pylearn2MLPO()
        self.target_network = self.network
        self.gamma = 0.95
        self.epsilon = 0.1
        print 'gamma', self.gamma, 'epsilon', self.epsilon, 'lr', learning_rate

        self.initialised = False

        self.memory = []
        self.last_sa = None
        self.last_s = None
        self.last_q = None

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
        if(self.swap_counter % self.swap_iterations ==0 ):
            pass

    def act_qs(self, state, reward, terminal, n_actions, q_filter):
        if not self.initialised:
            sa = state.reshape(1, state.size)
            target = np.zeros((1, n_actions))
            self.network.linit(sa, target)
            self.initialised = True

        qs = self.__Qs(np.array([state]))[0]
        if np.random.random() < self.epsilon:
            action = np.random.randint(0, n_actions)
        else:
            action = (qs * q_filter).argmax()

        last_s, last_q = self.last_s, self.last_q
        if last_s is not None:
            rewards = np.zeros((n_actions,))
            rewards[action] = reward + qs[action] * 0.99
            self.addToMemory(last_s, (last_q+rewards).clip(-1.0, 1.0), terminal, state)
        self.last_s = state
        self.last_q = qs
        return action

    def train_qs(self, n_updates, n_last=0, latest_ratio=0.15):
        updates = min(len(self.memory), n_updates)
        inputs = np.zeros((updates, self.memory[0][0].size))
        targets = np.zeros((updates, self.memory[0][1].size))

        if len(self.memory)-n_last == 0:
            latest_ratio = 1.0

        n_latest, n_random = 0, 0
        for i in range(updates):
            if np.random.random() < latest_ratio:                    
                r = np.random.randint(len(self.memory)-n_last, len(self.memory))
                n_latest += 1
            else:
                r = np.random.randint(len(self.memory)-n_last)
                n_random += 1
            sample = self.memory[r]
            inputs[i] = sample[0]
            targets[i] = sample[1]
        print "  - samples %iR / %iL" % (n_random, n_latest)
        self.network.fit(inputs, targets, epochs=10)
        predicted = self.network.predict(inputs)
        print "  - error %f" % ((targets - predicted) ** 2).mean()
        print "  - predicted %f / %f / %f" % (predicted.min(), predicted.mean(), predicted.max())
        print "  - targets %f / %f / %f" % (targets.min(), targets.mean(), targets.max())

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

