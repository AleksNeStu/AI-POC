from model.game_state import GameState
from websocket_handler import WebSocketHandler

class GameController:
    def __init__(self, socketio):
        self.game_state = GameState()
        self.websocket_handler = WebSocketHandler(socketio, self.game_state)
        self.websocket_handler.start_game_loop()

    def update_game(self):
        # Logic to update the game state
        self.game_state.update()
        self.websocket_handler.broadcast_state()