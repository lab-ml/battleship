from typing import Dict, Union

import torch

from labml import logger
from labml.logger import Color


class Board:
    EMPTY: int = 0
    SHIP: int = 1
    BOMBED: int = 2

    ROW = 'r'
    COLUMN = 'c'

    LETTERS: Dict = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'J': 9}

    SHIPS: Dict = {'destroyer': 2, 'submarine': 2, 'cruiser': 3, 'battleship': 4, 'carrier': 5}

    def __init__(self, ships: Dict):
        self.board = torch.zeros(10, 10, dtype=torch.int)
        self.ships = ships

        self._initialize()

        self.sunk_ships = []

    def _initialize(self):
        for ship, (let, num, kind) in self.ships.items():
            assert ship in self.SHIPS
            assert kind in [self.ROW, self.COLUMN]

            let, num = self._validate(let, num)

            size = self.SHIPS[ship]

            if kind == self.ROW:
                self.board[num, let:let + size] = self.SHIP
            else:
                self.board[num: num + size, let] = self.SHIP

    def _validate(self, let: Union[str, int], num: int):
        if type(let) == str:
            assert let in self.LETTERS
            let = self.LETTERS[let]
        else:
            assert 0 <= let <= 9

        assert 0 <= num <= 9

        return let, num

    def play(self, let: Union[str, int], num: int):
        let, num = self._validate(let, num)

        square = self.board[num, let]

        if square == self.EMPTY:
            logger.log('you missed my battleships!', Color.green)
        elif square == self.SHIP:
            self.board[num, let] = self.BOMBED
            logger.log('you bombed my ship!', Color.orange)
        else:
            raise ValueError('you guessed that one already')

    def is_sunk_ship(self):
        for ship, (let, num, kind) in self.ships.items():
            if ship in self.sunk_ships:
                continue

            let, num = self._validate(let, num)

            size = self.SHIPS[ship]

            if kind == self.ROW:
                ship_sum = self.board[num, let:let + size].sum()
            else:
                ship_sum = self.board[num: num + size, let].sum()

            if ship_sum == size * self.BOMBED:
                self.sunk_ships.append(ship)
                logger.log('congratulations! you sunk my {}'.format(ship), Color.purple)
                return True

        return False

    def print_board(self):
        from prettytable import PrettyTable

        x = PrettyTable(hrules=True)
        columns = [key for key in self.LETTERS]
        columns.insert(0, '')

        x.field_names = columns

        for i, row in enumerate(self.board):
            row = row.tolist()
            row.insert(0, i)
            x.add_row(row)

        logger.log(x.get_string(), Color.blue)

    def is_won(self):
        if len(self.sunk_ships) == len(self.SHIPS):
            logger.log('congratulations! you sunk my every ship', Color.red)
            return True

        return False
