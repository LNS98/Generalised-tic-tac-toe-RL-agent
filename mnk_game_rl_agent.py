"""
Implementation of hte m-n-k board game.
"""

import random
import time
from player import *

class Game:

    def __init__(self, m, n, k):
        self.m = m  # horizontal length of the board
        self.n = n  # vertical legnth of board
        self.k = k  # number of things in a row
        # empty board
        self.board = Board(self.m, self.n, self.k)
        self.board.initi_board()


    def initialize_game(self, player1, player2):

        # init the two players
        self.player1 = player1 # PlayerRL(self.board, "X", self.m, self.n, self.k)
        self.player2 = player2 # PlayerRL(self.board, "O", self.m, self.n, self.k)

        #  print the board
        # self.board.drawboard()

    def reset_game(self):
        # empty board
        self.board = Board(self.m, self.n, self.k)
        self.board.initi_board()
        # re initialse the scores
        self.scores = [0 for i in range(self.m + self.n + 2*(self.m+self.n-1))]
        self.positions_occcupied = 0

        # make the board be the one for the agent
        self.player1.board = self.board
        self.player2.board = self.board



    def game_status(self):
        """
        Check the board status to see who (if anyone) has won
        """
        # get the result from the  board  status function
        status = self.board.board_status()

        # player 1 has won
        if status == 1:
            print("Game finished, player 1 has won")

        # player 2 has won
        if status == -1:
            print("Game finished, player 2 has won")

        if status == 0:
            print("Game finished, game tied")


        return status


    def play(self, debug = False):

        # start clock
        # start = time.time()

        episode_states = []

        # while not game finished
        while True:
            for player in [self.player1, self.player2]:


                # make the move for the players
                row, col = player.move()
                # change the baord
                self.board._add_move(row, col, player.name_player)
                # show the board
                # make the game go slower
                if debug == True:
                    time.sleep(0.5)
                    self.board.drawboard()

                # check the status of the game
                status = self.game_status()

                # append the state to the episodes states
                episode_states.append(self.board.scores)

                # evaluate the speed at every move
                # time_per_move = time.time() - start
                # start = time.time()
                # print("-------- time per move: {}".format(time_per_move))

                # if status is true the game has ended
                if status != None:
                    # reset the game
                    self.reset_game()

                    return (episode_states, status)


class Board:

    def __init__(self, m, n, k):
        self.m = m
        self.n = n
        self.k = k

        # used to keep track of who has won or if the game is finished
        self.scores = [0 for i in range(m + n + 2*(m+n-1))]
        self.positions_occcupied = 0


    def initi_board(self):
        # empty board
        array_board = [[" " for i in range(self.n)] for j in range(self.m)]
        self.array_board = array_board

        return array_board

    def drawboard(self):
        list_lines = []


        for index_line, array_line in enumerate(self.array_board):
            number_spaces_before_line = 2 - len(str(index_line))
            space_before_line = number_spaces_before_line * ' '
            list_lines.append(f'{space_before_line}{index_line}|  ' + '  |  '.join(array_line) + '  |\n')

        line_dashes = '   ' + '-' * 6 * self.n + '-\n'

        board_str = line_dashes + line_dashes.join(list_lines) + line_dashes

        print(board_str)


    def is_valid(self, row, col):

        return  ((0 <= row <= self.m and 0 <= col <= self.n) and
                (self.array_board[row][col] == " "))


    def _add_move(self, row, col, player_name):
        # change the posiiton of the array_board
        self.array_board[row][col] = player_name

        # update the posiitons of the scores to keep track of who has won

        # get the mark to add depending on the player
        if player_name == "X":
            mark = 1
        else:
            mark = -1

        self.scores[row] += mark # add +/-1 to the row index
        self.scores[self.m + col] += mark # add +/-1 to the col index
        self.scores[self.m + self.n + row + col] += mark  # add +/-1 to the diag index
        self.scores[self.m + self.n + (self.m+self.n-2) + (self.m - row) + col] += mark  # add +/-1 to the other diag index

        # add 1 to the palces occupied
        self.positions_occcupied += 1

    def _undo_move(self, row, col, player_name):

        # change the posiiton of the array_board
        self.array_board[row][col] = " "

        # update the posiitons of the scores to keep track of who has won

        # get the mark to add depending on the player
        if player_name == "X":
            mark = 1
        else:
            mark = -1

        self.scores[row] -= mark # add +/-1 to the row index
        self.scores[self.m + col] -= mark # add +/-1 to the col index
        self.scores[self.m + self.n + row + col] -= mark  # add +/-1 to the diag index
        self.scores[self.m + self.n + (self.m+self.n-2) + (self.m - row) + col] -= mark  # add +/-1 to the other diag index

        # add 1 to the palces occupied
        self.positions_occcupied -= 1



    def board_status(self):
        """
        Check if the board is in a finished state.
        e.g. either full or someone has won
        """

        # check if any values of the score are +/- k
        if self.k in self.scores:
            return 1
        if -self.k in self.scores:
            return -1

        # check if board is full
        if self.positions_occcupied == self.m * self.n:
            return 0

        # otherwise keep playing
        return None




if __name__ == '__main__':

    # init define variables for the game
    m, n, k = 3, 3, 3
    # init class
    dummy_game = Game(m, n, k)
    # initialise game
    dummy_game.initialize_game()

    dummy_game.play()
