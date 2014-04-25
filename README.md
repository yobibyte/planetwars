planetwars-server
==========

## Packages and Modules

	sudo apt-get install python2.7-dev python-virtualenv libevent-dev
	pip install flask gevent gevent-socketio werkzeug jinja2 itsdangerous gevent socketIO-server

## Starting the Flask Server

	# virtualenv is highly recommended.
	virtualenv env
	# Add the parent directory to the PYTHONPATH.
	echo $PARENT_DIR > env/lib/python2.7/site-packages/www.pth
	# Activate our virtualenv.
	. env/bin/activate
	# Run the server!
	python web/server.py
