from threading import Thread

from flask import Flask, request, redirect, url_for, send_from_directory
from flask_socketio import SocketIO, emit
import os

dir_path = os.path.dirname(os.path.realpath(__file__))


class Webserver:
    def __init__(self):
        # Setup Flask app.
        app = Flask(__name__)
        app.config['SECRET_KEY'] = "persittpassord"
        socketio = SocketIO(app)
        self.socketio = socketio
        app.debug = False

        # Routes
        @app.route('/')
        def root():
            return send_from_directory(os.path.join(dir_path, 'public_html'), 'index.html')

        @app.route('/<path:path>')
        def static_proxy(path):
            # send_static_file will guess the correct MIME type
            return send_from_directory(os.path.join(dir_path, 'public_html'), path)

        @socketio.on('my event', namespace='/test')
        def test_message(message):
            emit('my response', {'data': message['data']})

        @socketio.on('connect')
        def test_connect():
            emit('connect', {'data': 'Connected'})

        @socketio.on('disconnect')
        def test_disconnect():
            print('Client disconnected')

        t = Thread(target=socketio.run, args=(app, '0.0.0.0', 8080))
        t.start()

    def emit(self, event,  data):
        self.socketio.emit(event, data)

