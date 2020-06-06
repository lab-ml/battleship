from typing import Union
from copy import deepcopy

import torch

from labml import logger
from labml.logger import Color

from battleship.consts import ROW, COLUMN, LETTERS, SHIPS, BOARD_SIZE, EMPTY, BOMBED, SHIP


class Board:
    def __init__(self, ships: Union[dict, torch.Tensor]):
        self._ships = ships
        self._sunk_ships = []

        if type(self._ships) == dict:
            self._board = self._generate_init_board()
        else:
            self._board = self._ships

        self._init_board = deepcopy(self._board)

    def _generate_init_board(self):
        board = torch.zeros(BOARD_SIZE, BOARD_SIZE, dtype=torch.int)
        for ship, (let, num, kind) in self._ships.items():
            assert ship in SHIPS
            assert kind in [ROW, COLUMN]

            let, num = self._validate(let, num)

            size = SHIPS[ship]

            if kind == ROW:
                board[num, let:let + size] = SHIP
            else:
                board[num: num + size, let] = SHIP

        return board

    @staticmethod
    def _validate(let: Union[str, int], num: int):
        if type(let) == str:
            assert let in LETTERS
            let = LETTERS[let]
        else:
            assert 0 <= let <= 9

        assert 0 <= num <= 9

        return let, num

    def play(self, let: Union[str, int], num: int):
        let, num = self._validate(let, num)

        square = self._board[num, let]

        if square == EMPTY:
            square.fill_(BOMBED)
            logger.log('you missed my battleships!', Color.green)
            return EMPTY
        elif square == SHIP:
            square.fill_(BOMBED)
            logger.log('you bombed my ship!', Color.red)
            return SHIP
        else:
            logger.log('wrong attempt!', Color.orange)
            return BOMBED

    def is_sunk_ship(self):
        for ship, (let, num, kind) in self._ships.items():
            if ship in self._sunk_ships:
                continue

            let, num = self._validate(let, num)

            size = SHIPS[ship]

            if kind == ROW:
                ship_sum = self._board[num, let:let + size].sum()
            else:
                ship_sum = self._board[num: num + size, let].sum()

            if ship_sum == size * BOMBED:
                self._sunk_ships.append(ship)
                logger.log('congratulations! you sunk my {}'.format(ship), Color.purple)
                return True

        return False

    def is_won(self):
        if len(self._sunk_ships) == len(SHIPS):
            logger.log('congratulations! you sunk my every ship', Color.red)
            return True

        return False

    def render_board(self):
        from prettytable import PrettyTable

        board = PrettyTable(hrules=True)
        columns = [key for key in LETTERS]
        columns.insert(0, '')

        board.field_names = columns

        for i, row in enumerate(self._board):
            row = row.tolist()
            row.insert(0, i)
            board.add_row(row)

        logger.log(board.get_string(), Color.blue)

    def get_current_board(self):
        return self._board

    def get_initial_board(self):
        return self._init_board
