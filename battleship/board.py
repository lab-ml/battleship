from typing import Union

import torch

from labml import logger
from labml.logger import Color

from battleship.consts import ROW, COLUMN, LETTERS, SHIPS, BOARD_SIZE, EMPTY, BOMBED, SHIP


class Board:
    def __init__(self, ships: Union[dict, torch.Tensor]):
        self.ships = ships
        self.sunk_ships = []

        if type(self.ships) == dict:
            self.board = self._generate_init_board()
        else:
            self.board = self.ships

    def _generate_init_board(self):
        board = torch.zeros(BOARD_SIZE, BOARD_SIZE, dtype=torch.int)
        for ship, (let, num, kind) in self.ships.items():
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

        square = self.board[num, let]

        if square == EMPTY:
            logger.log('you missed my battleships!', Color.green)
            return EMPTY
        elif square == SHIP:
            square.fill_(BOMBED)
            logger.log('you bombed my ship!', Color.orange)
            return SHIP
        else:
            logger.log('wrong attempt!', Color.orange)
            return BOMBED

    def is_sunk_ship(self):
        for ship, (let, num, kind) in self.ships.items():
            if ship in self.sunk_ships:
                continue

            let, num = self._validate(let, num)

            size = SHIPS[ship]

            if kind == ROW:
                ship_sum = self.board[num, let:let + size].sum()
            else:
                ship_sum = self.board[num: num + size, let].sum()

            if ship_sum == size * BOMBED:
                self.sunk_ships.append(ship)
                logger.log('congratulations! you sunk my {}'.format(ship), Color.purple)
                return True

        return False

    def render_board(self):
        from prettytable import PrettyTable

        board = PrettyTable(hrules=True)
        columns = [key for key in LETTERS]
        columns.insert(0, '')

        board.field_names = columns

        for i, row in enumerate(self.board):
            row = row.tolist()
            row.insert(0, i)
            board.add_row(row)

        logger.log(board.get_string(), Color.blue)

    def is_won(self):
        if len(self.sunk_ships) == len(SHIPS):
            logger.log('congratulations! you sunk my every ship', Color.red)
            return True

        return False
