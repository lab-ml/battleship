from typing import List, Union

import random
import itertools

from labml import logger
from labml.logger import Color

from battleship.board import Board


def generate_random_attacks(numbers: Union[List, range]):
    pairs = list(itertools.permutations(numbers, 2))
    pairs += ([(i, i) for i in numbers])

    random.shuffle(pairs)

    return pairs


def random_game():
    inputs = {'destroyer': ('C', 2, 'r'), 'submarine': ('F', 3, 'c'), 'cruiser': ('B', 5, 'r'),
              'battleship': ('F', 6, 'c'), 'carrier': ('F', 1, 'r')}

    game = Board(inputs)

    print(game.board)

    for attempt, attack in enumerate(generate_random_attacks(range(10))):
        logger.log('attempt : ' + str(attempt), Color.cyan)

        game.play(attack[0], attack[1])
        game.is_sunk_ship()

        if game.is_won():
            break

    print(game.board)


if __name__ == '__main__':
    random_game()
