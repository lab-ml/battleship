import torch

import random
from random import randint

from labml import monit

from battleship.consts import ROW, COLUMN, SHIPS, BOARD_SIZE

KINDS = [ROW, COLUMN]


def generate_ships(n_games: int):
    for _ in range(n_games):
        board = torch.zeros(BOARD_SIZE, BOARD_SIZE, dtype=torch.int)

        for ship, size in SHIPS.items():
            is_found = False
            while not is_found:
                kind = KINDS[randint(0, 1)]

                rand_nums = [random.choice(range(10 - size)), random.choice(range(10))]
                if kind == ROW:
                    let, num = rand_nums[0], rand_nums[1]
                    prop = board[num, let:let + size]
                else:
                    let, num = rand_nums[1], rand_nums[0]
                    prop = board[num: num + size, let]

                if prop.sum() == 0:
                    is_found = True
                    prop.fill_(1)

        yield board


def test():
    with monit.section("generate"):
        for config in generate_ships(100000):
            if not config.sum() == 16:
                raise ValueError


if __name__ == '__main__':
    test()
