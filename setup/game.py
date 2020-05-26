"""
Game class used to play to m, n, k game.
"""

import time 

from setup.board import Board


class Game:

    def __init__(self, m, n, k, display=True):
        self.m = m  # horizontal length of the board
        self.n = n  # vertical legnth of board
        self.k = k  # number of things in a row
        self.display = display
        # empty board
        self.board = Board(self.m, self.n, self.k)
        self.board.initi_board()


    def initialize_game(self, player1, player2):

        # init the two players
        self.player1 = player1 
        self.player2 = player2
        
        # splecify the two players to have the same board
        self.player1.board = self.board
        self.player2.board = self.board
        
        #  print the board
        if self.display:
            self.board.drawboard()

    def reset_game(self):
        # empty board
        self.board = Board(self.m, self.n, self.k)
        self.board.initi_board()
#        # re initialse the scores
#        self.scores = [0 for i in range(self.m + self.n + 2*(self.m+self.n-1))]
#        self.positions_occcupied = 0
#
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


    def play(self):

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
                if self.display == True:
                    time.sleep(0.5)
                    self.board.drawboard()

                # check the status of the game
                status = self.game_status()

                # append the state to the episodes states
                episode_states.append(self.board.scores)

                # if status is true the game has ended
                if status != None:
                    # reset the game
                    self.reset_game()

                    return (episode_states, status)


if __name__ == "__main__":

    dummy_game = Game(3, 3, 3)

    print(dummy_game)
