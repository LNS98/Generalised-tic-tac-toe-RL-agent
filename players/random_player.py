"""
Implemenatation for  a random agent which will play the game.
"""
import random

from players.base_player import Player

class PlayerRandom(Player):
    """
    Random player that playes randomly
    """

    def __init__(self, board, name_player, m, n, k):

        super().__init__(board, name_player, m, n, k)

    def move(self):

        # choose a position ranadomly
        row = random.randint(0, self.m-1)
        col = random.randint(0, self.n-1)

        # check if the cordintaes are valid
        while not self.board.is_valid(row, col):

            # get new ones if not valid
            row = random.randint(0, self.m-1)
            col = random.randint(0, self.n-1)

        return (row, col)

