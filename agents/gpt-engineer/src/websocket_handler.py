from flask_socketio import SocketIO
from model.game_state import GameState

class WebSocketHandler:
    def __init__(self, socketio: SocketIO, game_state: GameState):
        self.socketio = socketio
        self.game_state = game_state

    def start_game_loop(self):
        # Start the game loop to update and broadcast the game state
        self.socketio.start_background_task(self.game_loop)

    def game_loop(self):
        while True:
            self.game_state.update()
            self.broadcast_state()
            self.socketio.sleep(0.1)  # Adjust the sleep time for game speed

    def broadcast_state(self):
        # Broadcast the current game state to all connected clients
        self.socketio.emit('game_state', self.game_state)