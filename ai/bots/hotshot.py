import random

from .. import planetwars_class
from planetwars.datatypes import Order
from planetwars.utils import *

KEEP_LIMIT = 0.078
    
NEUTRAL_CLOSEST = 0.062
NEUTRAL_HIGHGROWTH = 0.047
NEUTRAL_LOWGROWTH = 0.013
NEUTRAL_ANY = 0.000
NEUTRAL_EASY = 0.108
    
THEIR_CLOSEST = 0.121
THEIR_HIGHGROWTH = 0.044
THEIR_LOWGROWTH = 0.226
THEIR_ANY = 0.000
THEIR_EASY = 0.035
    
MAX_IN_FLIGHT = 0.310
LEAVE_ON_PLANET = 0.036

@planetwars_class
class Hotshot( object ):

    def __init__( self ):
        self.hotshot_params = [KEEP_LIMIT, NEUTRAL_CLOSEST, NEUTRAL_HIGHGROWTH, NEUTRAL_LOWGROWTH, NEUTRAL_ANY, NEUTRAL_EASY, THEIR_CLOSEST, THEIR_HIGHGROWTH, THEIR_LOWGROWTH, THEIR_ANY, THEIR_EASY, MAX_IN_FLIGHT, LEAVE_ON_PLANET]
        '''
        self.previous_params = self.hotshot_params[:]
        self.turns = 0
        self.previous_fitness = 0
        self.fitness = 0
        
    def done( self, won ):
        if won:
            self.fitness += 1
        self.turns += 1
        if self.turns%100 == 0:
            if self.fitness >= self.previous_fitness:
                self.previous_params = self.hotshot_params[:]
                self.previous_fitness = self.fitness
            else:
                self.hotshot_params = self.previous_params[:]
            print self.previous_fitness
            print self.previous_params
            self.fitness = 0
            i = 0
            while i < len( self.hotshot_params ):
                if random.random() < 0.25:
                    self.hotshot_params[i] *= 0.7 +0.6*random.random()
                    if self.hotshot_params[i] > 1:
                        self.hotshot_params[i] = 1
                i += 1
    '''
    def __call__( self, turn, pid, planets, fleets ):
        
        i = 1
        sumparams = 0
        while i < 11:
            sumparams += self.hotshot_params[i]
            i += 1
        if sumparams > 1 - self.hotshot_params[12]:
            #print "adapt params", sumparams, self.hotshot_params[12]
            factor = (1 - self.hotshot_params[12]) / sumparams
            i = 1
            while i < 11:
                self.hotshot_params[i] *= factor
                i += 1
                
        my_planets, their_planets, neutral_planets = aggro_partition( pid, planets )
        orders = [] 

        if random.random() < 0.15 and False:
            other_planets = their_planets+neutral_planets
            if not my_planets or not other_planets:
                return []
            source = random.choice(my_planets)
            destination = random.choice(other_planets)
            return [Order(source, destination, source.ships / 2)]

        docked = 0
        flying = 0
        for planet in my_planets:
            docked += planet.ships
        for fleet in fleets:
            if fleet.owner == pid:
                flying += fleet.ships
        if flying + docked > 0: 
            if float( flying ) / float( flying + docked ) > self.hotshot_params[11]:
                return []

        for planet in my_planets:
            if planet.ships < self.hotshot_params[0]*100:
                continue
            def dist_to( other_planet ):
                return turn_dist( planet, other_planet )
            def highgrowth_of( other_planet ):
                return other_planet.growth * 1000 - turn_dist( planet, other_planet )
            def lowgrowth_of( other_planet ):
                return other_planet.growth * 1000 + turn_dist( planet, other_planet )
            def ease_of( other_planet ):
                return other_planet.ships * 1000 + turn_dist( planet, other_planet )
            if len( neutral_planets ) > 0:
                if self.hotshot_params[1] > 0:
                    closest = min( neutral_planets, key=dist_to )       
                    orders.append( Order( planet, closest, planet.ships * self.hotshot_params[1] ) )
                if self.hotshot_params[2] > 0:
                    high_growth = max( neutral_planets, key=highgrowth_of )       
                    orders.append( Order( planet, high_growth, planet.ships * self.hotshot_params[2] ) )
                if self.hotshot_params[3] > 0:
                    low_growth = min( neutral_planets, key=lowgrowth_of )       
                    orders.append( Order( planet, high_growth, planet.ships * self.hotshot_params[3] ) )
                if self.hotshot_params[4] > 0:
                    any = neutral_planets[random.randint( 0, len( neutral_planets )-1 )]       
                    orders.append( Order( planet, any, planet.ships * self.hotshot_params[4] ) )
                if self.hotshot_params[5] > 0:
                    easy = min( neutral_planets, key=ease_of )       
                    orders.append( Order( planet, easy, planet.ships * self.hotshot_params[5] ) )
            if len( their_planets ) > 0:
                if self.hotshot_params[6] > 0:
                    closest = min( their_planets, key=dist_to )       
                    orders.append( Order( planet, closest, planet.ships * self.hotshot_params[6] ) )
                if self.hotshot_params[7] > 0:
                    high_growth = max( their_planets, key=highgrowth_of )       
                    orders.append( Order( planet, high_growth, planet.ships * self.hotshot_params[7] ) )
                if self.hotshot_params[8] > 0:
                    low_growth = min( their_planets, key=lowgrowth_of )       
                    orders.append( Order( planet, high_growth, planet.ships * self.hotshot_params[8] ) )
                if self.hotshot_params[9] > 0:
                    any = their_planets[random.randint( 0, len( their_planets )-1 )]       
                    orders.append( Order( planet, any, planet.ships * self.hotshot_params[9] ) )
                if self.hotshot_params[10] > 0:
                    easy = min( their_planets, key=ease_of )       
                    orders.append( Order( planet, easy, planet.ships * self.hotshot_params[10] ) )

        return orders





'''

def hotshot_02( turn, pid, planets, fleets ):

    def target_value( source, target, fleets ):
        
        distance = turn_dist( source, target )
        growthrate = target.growth
        my_underway = 0
        their_underway = 0
        for fleet in fleets:
            if fleet.destination == target:
                if fleet.owner == source.owner:
                    my_underway += fleet.ships
                else:
                    their_underway = fleet.ships
        if target.owner == source.owner:
            my_underway += target.ships
        elif target.owner != 0:
            their_underway += target.ships

        if my_underway > their_underway:
            return 0
        return (1.0 / distance)**2 * growthrate

    my_planets, their_planets, neutral_planets = aggro_partition( pid, planets )
    orders = [] 

    for planet in my_planets:
        if planet.ships < 10:
            continue
        def value( other_planet ):
            return target_value( planet, other_planet, fleets )
        other_planets = their_planets + neutral_planets
        maxvalue = max( other_planets, key=value )       
        orders.append( Order( planet, maxvalue, planet.ships * 0.25 ) )

    return orders







def hotshot_01( turn, pid, planets, fleets ):

    my_planets, their_planets, neutral_planets = aggro_partition( pid, planets )
    orders = [] 

    for planet in my_planets:
        if planet.ships < 20:
            continue
        def dist_to( other_planet ):
            return turn_dist( planet, other_planet )
        def growth_of( other_planet ):
            return other_planet.growth
        if len( neutral_planets ) > 0:
            closest = min( neutral_planets, key=dist_to )       
            orders.append( Order( planet, closest, planet.ships * 0.1 ) )
            high_growth = max( neutral_planets, key=growth_of )       
            orders.append( Order( planet, high_growth, planet.ships * 0.2 ) )
            low_growth = min( neutral_planets, key=growth_of )       
            orders.append( Order( planet, high_growth, planet.ships * 0.05 ) )
            any = neutral_planets[random.randint( 0, len( neutral_planets )-1 )]       
            orders.append( Order( planet, any, planet.ships * 0.05 ) )
        if len( their_planets ) > 0:
            closest = min( their_planets, key=dist_to )       
            orders.append( Order( planet, closest, planet.ships * 0.1 ) )
            high_growth = max( their_planets, key=growth_of )       
            orders.append( Order( planet, high_growth, planet.ships * 0.05 ) )
            low_growth = min( their_planets, key=growth_of )       
            orders.append( Order( planet, high_growth, planet.ships * 0.2 ) )
            any = their_planets[random.randint( 0, len( their_planets )-1 )]       
            orders.append( Order( planet, any, planet.ships * 0.05 ) )

    return orders
'''