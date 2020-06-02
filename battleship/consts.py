from typing import Dict

ROW = 'r'
COLUMN = 'c'

LETTERS: Dict = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'J': 9}

SHIPS: Dict = {'destroyer': 2, 'submarine': 2, 'cruiser': 3, 'battleship': 4, 'carrier': 5}

BOARD_SIZE = 10

EMPTY: int = 0
SHIP: int = 1
BOMBED: int = 2
WON: int = 3
SUNK_SHIP: int = 4


def num_to_let(num: int):
    for _let, _num in LETTERS.items():
        if num == _num:
            return _let
