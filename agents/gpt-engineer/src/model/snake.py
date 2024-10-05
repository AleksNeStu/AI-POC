from dataclasses import dataclass, field
from typing import List, Tuple

@dataclass
class Snake:
    body: List[Tuple[int, int]] = field(default_factory=list)
    direction: Tuple[int, int] = (0, 1)

    def move(self):
        # Logic to move the snake in the current direction
        new_head = (self.body[0][0] + self.direction[0], self.body[0][1] + self.direction[1])
        self.body = [new_head] + self.body[:-1]