from typing import List, Union, Optional

import random
import itertools

from labml import logger
from labml.logger import Color

from battleship.board import Board


class RandomAgent:
    STATE_SIZE = 10

    def __init__(self):
        self._iterator = -1
        self.random_attacks = self.generate_random_attacks(range(self.STATE_SIZE))

    def get_action(self, state=None):
        self._iterator += 1
        return self.random_attacks[self._iterator]

    def get_iterator(self):
        return self._iterator

    @staticmethod
    def generate_random_attacks(numbers: Union[List, range]):
        pairs = list(itertools.permutations(numbers, 2))
        pairs += ([(i, i) for i in numbers])

        random.shuffle(pairs)

        return pairs


def random_game():
    inputs = {'destroyer': ('C', 2, 'r'), 'submarine': ('F', 3, 'c'), 'cruiser': ('B', 5, 'r'),
              'battleship': ('F', 6, 'c'), 'carrier': ('F', 1, 'r')}

    game = Board(inputs)

    game.render_board()

    agent = RandomAgent()
    while not game.is_won():
        r, c = agent.get_action()

        logger.log('attempt : ' + str(agent.get_iterator()), Color.cyan)

        game.play(r, c)
        game.is_sunk_ship()

    game.render_board()


if __name__ == '__main__':
    random_game()
