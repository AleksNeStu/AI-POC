from dataclasses import dataclass, field
from typing import List
from model.snake import Snake

@dataclass
class GameState:
    snakes: List[Snake] = field(default_factory=list)
    food_position: tuple = (0, 0)

    def update(self):
        # Update the game state, move snakes, check collisions, etc.
        for snake in self.snakes:
            snake.move()
        # Additional logic to handle food and collisions