__author__ = 'ssamot, schaul'

from ai.bots.nn.sknn.sknn import sknn, IncrementalMinMaxScaler
import numpy as np
import cPickle as pickle



class DeepQ():
    """
    A Q learning agent
    """

    def __init__(self, layers,  dropout = False, input_scaler=IncrementalMinMaxScaler(), output_scaler=IncrementalMinMaxScaler(),   learning_rate=0.005, verbose=0):
        self.max_memory = 500000
        self.memory = []
        self.network = sknn(layers, dropout, None, None, learning_rate,verbose)
        ##self.target_network = pylearn2MLPO()
        self.target_network = self.network
        self.gamma = 0.9
        self.epsilon = 0.5
        print 'gamma', self.gamma, 'epsilon', self.epsilon, 'lr', learning_rate
        self.swap_iterations = 10000
        self.swap_counter = 0

        self.initialised = False

        self.memory = []
        self.last_sa = None

    def addToMemory(self, last_sa, reward, terminal, all_next_sas):
        self.memory+=[(last_sa,reward,terminal,all_next_sas)]

    def train_from_memory(self, updates):

        if len(self.memory) > 1000:
          updates = min(len(self.memory), updates)
          inputs = np.zeros((updates, self.memory[0][0].size))
          targets = np.zeros((updates, 1))
          for update in range(updates):

            r = np.random.randint(len(self.memory))
            input = self.memory[r]
            target = self.computeTarget(*input)

            inputs[update] = input[0]
            targets[update] = target

          self.network.fit(inputs, targets)
        else:
           print "not enough"
        if len(self.memory) > self.max_memory:
           print "clipped memory"
           self.memory = self.memory[self.max_memory/2:]




    def __Qs(self,sas):
        #Q = np.array([self.target_network.predict(state_action.reshape(1,state_action.size) )for state_action in sas])
        Q = self.target_network.predict(sas)
        return Q

    def computeTarget(self, last_sa, reward, terminal, all_next_sas):
        #last_sa = self.last_sa
        gamma = self.gamma
        maxQ = 0
        if terminal == 0:
            maxQ = self.__Qs(all_next_sas).max()

        target = reward  + (1-terminal) * gamma * maxQ

        return target


    def fit(self,last_sa, reward, terminal, all_next_sas):
        #last_sa = self.last_sa
        gamma = self.gamma
        maxQ = 0
        if terminal == 0:
            maxQ = self.__Qs(all_next_sas).max()

        target = reward  + (1-terminal) * gamma * maxQ
        self.network.fit(last_sa.reshape(1,last_sa.size), np.array([[target]]))
        if(self.swap_counter % self.swap_iterations ==0 ):
            pass



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

