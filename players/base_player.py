"""
Base class used for more different types of player, but a human player can also play using this class
"""

from setup.game import Game 


class Player(Game):

    def __init__(self, board, name_player, m, n, k):

        self.board = board
        self.name_player = name_player
        super().__init__(m, n, k)

    def move(self):

        # player 1  place a thing on the coordinate
        row = int(input("Row number: "))
        col = int(input("column number: "))

        # ADD SOME CHECKS
        while not self.board.is_valid(row, col):

            print("Inputs not valid.")
            print("Please provide a row number between {} and {}".format(0, self.m))
            print("Please provide a column number between {} and {}".format(0, self.n))
            print("Also make sure that the entry is empty.")

            # re-ask for the inputs
            row = int(input("Row number: "))
            col = int(input("column number: "))

        return (row, col)


