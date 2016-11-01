Planet Wars Experiments
=======================

## Important!

If you cannot run a game server due to unicode error, run this:

```bash
pip2 install -U gevent==1.1b4
```
## Model

Best model available (~70% vs Evolved) is [here](https://dl.dropboxusercontent.com/u/23750836/model.h5).

## Description (taken from [google ai challenge](https://web.archive.org/web/20120510223543/http://planetwars.aichallenge.org/problem_description.php))

The objective is to create a computer program that plays the game of Planet Wars as intelligently as possible. Planet Wars is a strategy game set in outer space. The objective is to take over all the planets on the map, or altenatively eliminate all of your opponents ships.

The game is turn-based. Your bot is a function that takes a list of planets and a list of fleets, and outputs some orders. Each planet has the following fields/properties:

* X-coordinate
* Y-coordinate
* Owner's PlayerID
* Number of Ships
* Growth Rate

Neutral planets have a PlayerID of zero, planets owned by your bot have a PlayerID of 1, and planets owned by the enemy have a PlayerID of 2. The number of ships and the growth rate are both whole numbers. Each turn, the number of ships on non-neutral planets increases according to the growth rate for that planet.

Fleets are the colored numbers that fly between planets. When a fleet arrives at its destination planet, one of two things can happen. If the destination planet belongs to your bot, the ships from the fleet land on the planet as reinforcements. If your bot does not own the destination planet, the ships from the fleet are subtracted from the ships that currently occupy the planet. If the result is less than zero, then your bot gains control of the planet. If the result is exactly zero, then control of the planet does not change hands. A fleet has the following properties:

* Owner's PlayerID
* Number of Ships
* Source PlanetID
* Destination PlanetID
* Total Trip Length
* Number of turns remaining until arrival

Your bot can issue as many orders as it wants during one turn. Each order specifies a source planet, a destination planet, and a number of ships. Once the order is executed, the given number of ships leave the source planet, to go towards the destination planet. Your bot issues orders using the IssueOrder(src, dest, num_ships) function. The game ends when only one player remains, or if the game goes past a certain number of turns.

More detailed information [here](https://web.archive.org/web/20120512101516/http://planetwars.aichallenge.org/specification.php).

## Running a Game (Console)
    python2.7 play.py Stochastic Stochastic

## Running a Competition (Console)
    python2.7 competition.py --games=100 Random Random

## Browser Visualization
First install dependencies:

    pip2.7 install gevent gevent-socketio flask

Then run the server itself:

    python2.7 web/server.py

Finally connect a browser to `http://127.0.0.1:4200/`.
