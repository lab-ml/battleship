from typing import Dict

import torch


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

    def _validate(self, let: str, num: int):
        assert let in self.LETTERS and 0 <= num <= 9

        return self.LETTERS[let], num

    def play(self, let: str, num: int):
        let, num = self._validate(let, num)

        square = self.board[num, let]

        if square == self.EMPTY:
            print('you missed my battleships!')
        elif square == self.SHIP:
            self.board[num, let] = self.BOMBED
            print('you bombed my ship!')
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
                print('congratulations! you sunk my {}'.format(ship))
                return True

        return False

    def is_won(self):
        if len(self.sunk_ships) == len(self.SHIPS):
            print('congratulations! you sunk my every ship')
            return True

        return False


class Game:
    pass


if __name__ == '__main__':
    inputs = {'destroyer': ('C', 2, 'r'), 'submarine': ('F', 3, 'c'), 'cruiser': ('B', 5, 'r'),
              'battleship': ('F', 6, 'c'), 'carrier': ('F', 1, 'r')}
    game = Board(inputs)

    print(game.board)
