import random
import copy

from math import log
from .. import planetwars_class
from planetwars.datatypes import Order, Planet, Fleet
from planetwars.utils import *
from planetwars import PlanetWars
from planetwars.views import TextView
from ai.state import State


@planetwars_class
class MCTS2(object):

    def __call__(self, turn, pid, planets, fleets):
        self.turn = turn
        self.pid = pid
        self.rootplanets = planets
        self.rootfleets = fleets

        pws = PlanetWarsState(turn, pid, planets, fleets)
        #print"Starting UCT - the Universe looks like:"
        #pws.PrintUniverse()

        return UCT(pws, 5, False)

    def PrintOrders(self):
        print ("Orders for player: " + str(self.pid))
        orders = self.state.generate_orders(self.pid)
        for o in orders:
            print o
        assert False

    def done(self, won):
        pass


class PlanetWarsState:
    """ A state of the game, i.e. the game board. These are the only functions which are
        absolutely necessary to implement UCT in any 2-player complete information deterministic 
        zero-sum game, although they can be enhanced and made quicker, for example by using a 
        GetRandomMove() function to generate a random move during rollout.
        By convention the players are numbered 1 and 2.
        """
    def __init__(self, turn, pid, planets, fleets):
        self.turn = turn
        self.pid = pid
        self.playerJustMoved = 3 - pid
        self.rootplanets = planets
        self.planets = [Planet(p.id, p.x, p.y, p.owner, p.ships, p.growth) for p in planets] # create mutable planets
        self.fleets = [Fleet(f.owner, f.ships, f.source, f.destination, f.total_turns, f.remaining_turns) for f in fleets] # create mutable fleets

        #for p in self.planets: print p
        #for f in fleets: print f
        #assert False
        
    def Clone(self):
        """ Create a deep clone of this game state.
        """
        turn = self.turn
        pid = self.pid
        planets = copy.deepcopy(self.planets)
        fleets = copy.deepcopy(self.fleets)
        return PlanetWarsState(turn, pid, planets, fleets)

    def DoMove(self, order):
        """ Update a state by carrying out the given move.
            Must update playerJustMoved.
        """
        self.playerJustMoved = 3 - self.playerJustMoved
        if order != None:         
            ships = int(min(order.ships, order.source.ships))
            order.source.ships -= ships
            self.fleets.append(Fleet(self.pid, ships, order.source, order.destination))
            
        for planet in self.planets:
            planet.generate_ships()

        for fleet in self.fleets:
           fleet.advance()

        self.playerJustMoved = self.pid
        self.pid = 3 - self.pid
           
        # Arrival
        arrived_fleets, self.fleets = partition(lambda fleet: fleet.has_arrived(), self.fleets)
        for planet in self.planets:
            planet.battle([fleet for fleet in arrived_fleets if fleet.destination == planet])

        
    def GetMoves(self):
        """ Get all possible moves from this state.
        """
        def mine(x):
          return x.owner == self.pid

        my_planets, other_planets = partition(mine, self.planets)
    
        res = []
     
        for src in my_planets:
          for dst in other_planets:
              if dst.ships < src.ships / 2:
                  res.append(Order(src, dst, src.ships / 2))

        if len(res) == 0 and len(my_planets) > 0 and len(other_planets) > 0:
            res.append(None)
##            src = max(my_planets, key=get_ships)
##            dst = min(other_planets, key=get_ships)
##            res.append(Order(src, dst, src.ships / 2))

        return res
    
    def GetResult(self, playerjm):
        """ Get the game result from the viewpoint of playerjm. 
        """
        P1_has_planets, P2_has_planets = False, False
        for p in self.planets:
            if p.owner == 1: P1_has_planets = True
            if p.owner == 2: P2_has_planets = True

        if P1_has_planets == False and P2_has_planets == False:
            return 0.5

        if P1_has_planets == False:
            if playerjm == 2: return 1
            else: return 0

        if P2_has_planets == False:
            if playerjm == 1: return 1
            else: return 0

        assert False, "Cannot have a game end if both players have planets"

    def GetOrderToPointToRootStateObjects(self, order):
        src = des = None

        for ps in self.rootplanets:
            if order.source.id == ps.id:
                src = ps

        for pd in self.rootplanets:
            if order.destination.id == pd.id:
                des = pd

        assert src is not None and des is not None, "Can't find planet in GetOrderToPointToRootStateObjects()"

        #self.PrintRootUniverse()
        #print Order(src, des, order.ships)
        
        return Order(src, des, order.ships)

    def PrintUniverse(self):
        tv = TextView()
        tv.update(self.planets, self.fleets)
        
    def PrintRootUniverse(self):
        tv = TextView()
        tv.update(self.rootplanets, self.fleets)
        
    def __repr__(self):
        """ Don't need this - but good style.
        """
        pass 

class Node:
    """ A node in the game tree. Note wins is always from the viewpoint of playerJustMoved.
        Crashes if state not specified.
    """
    def __init__(self, move = None, parent = None, state = None):
        self.move = move # the move that got us to this node - "None" for the root node
        self.parentNode = parent # "None" for the root node
        self.childNodes = []
        self.wins = 0
        self.visits = 0
        self.untriedMoves = state.GetMoves() # future child nodes
        self.playerJustMoved = state.playerJustMoved # the only part of the state that the Node needs later
        
    def UCTSelectChild(self):
        """ Use the UCB1 formula to select a child node. Often a constant UCTK is applied so we have
            lambda c: c.wins/c.visits + UCTK * sqrt(2*log(self.visits)/c.visits to vary the amount of
            exploration versus exploitation.
        """
        s = sorted(self.childNodes, key = lambda c: c.wins/c.visits + sqrt(2*log(self.visits)/c.visits))[-1]
        return s
    
    def AddChild(self, m, s):
        """ Remove m from untriedMoves and add a new child node for this move.
            Return the added child node
        """
        n = Node(move = m, parent = self, state = s)
        self.untriedMoves.remove(m)
        self.childNodes.append(n)
        return n
    
    def Update(self, result):
        """ Update this node - one additional visit and result additional wins. result must be from the viewpoint of playerJustmoved.
        """
        self.visits += 1
        self.wins += result

    def __repr__(self):
        return "[M:" + str(self.move) + " W/V:" + str(self.wins) + "/" + str(self.visits) + " U:" + str(self.untriedMoves) + "]"

    def TreeToString(self, indent):
        s = self.IndentString(indent) + str(self)
        for c in self.childNodes:
             s += c.TreeToString(indent+1)
        return s

    def IndentString(self,indent):
        s = "\n"
        for i in range (1,indent+1):
            s += "| "
        return s

    def ChildrenToString(self):
        s = ""
        for c in self.childNodes:
             s += str(c) + "\n"
        return s

def UCT(rootstate, itermax, verbose = False):
    """ Conduct a UCT search for itermax iterations starting from rootstate.
        Return the best move from the rootstate.
        Assumes 2 alternating players (player 1 starts), with game results in the range [0.0, 1.0]."""

    rootnode = Node(state = rootstate)

    for i in range(itermax):
        node = rootnode
        state = rootstate.Clone()

        # Select
        while node.untriedMoves == [] and node.childNodes != []: # node is fully expanded and non-terminal
            node = node.UCTSelectChild()
            state.DoMove(node.move)

        # Expand
        if node.untriedMoves != []: # if we can expand (i.e. state/node is non-terminal)
            m = random.choice(node.untriedMoves) 
            state.DoMove(m)
            node = node.AddChild(m,state) # add child and descend tree

        # Rollout - this can often be made orders of magnitude quicker using a state.GetRandomMove() function
        while state.GetMoves() != []: # while state is non-terminal
            state.DoMove(random.choice(state.GetMoves()))

        # Backpropagate
        while node != None: # backpropagate from the expanded node and work back to the root node
            node.Update(state.GetResult(node.playerJustMoved)) # state is terminal. Update node with result from POV of node.playerJustMoved
            node = node.parentNode

    # Output some information about the tree - can be omitted
    #if (verbose): print rootnode.TreeToString(0)
    #else: print rootnode.ChildrenToString()

    move = sorted(rootnode.childNodes, key = lambda c: c.visits)[-1].move # return the move that was most visited
    if move is None:
        #print "UCT thinks there are no sensible moves at all"
        return []
    rootmove = rootstate.GetOrderToPointToRootStateObjects(move)
    if rootmove.source.ships / 2 <= rootmove.destination.ships:
        #print "UCT gave a crappy answer - rejected"
        return []
    else:
        return [rootmove]
                                                                       



