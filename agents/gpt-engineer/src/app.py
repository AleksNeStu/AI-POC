from flask import Flask, render_template
from flask_socketio import SocketIO
from controller.game_controller import GameController

app = Flask(__name__)
socketio = SocketIO(app, async_mode='threading')

game_controller = GameController(socketio)

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000, allow_unsafe_werkzeug=True)