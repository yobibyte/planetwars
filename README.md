Planet Wars Experiments
=======================

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
