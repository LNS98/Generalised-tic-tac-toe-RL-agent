"""
Implementation of hte m-n-k board game.
"""

import random
import time

class Game:

    def __init__(self, m, n, k):
        self.m = m  # horizontal length of the board
        self.n = n  # vertical legnth of board
        self.k = k  # number of things in a row


    def initialize_game(self):
        # empty board
        self.board = Board(self.m, self.n, self.k)
        self.board.initi_board()

        # init the two players
        self.player1 = PlayerAI(self.board, "X", self.m, self.n, self.k)
        self.player2 = Player(self.board, "O", self.m, self.n, self.k)

        #  print the board
        self.board.drawboard()

    def game_status(self):
        """
        Check the board status to see who (if anyone) has won
        """
        # get the result from the  board  status function
        status = self.board.board_status()

        # player 1 has won
        if status == 1:
            status = True
            print("Game finished, player 1 has won")

        # player 2 has won
        if status == -1:
            status = True
            print("Game finished, player 2 has won")

        if status == 0:
            status = True
            print("Game finished, game tied")


        return status


    def play(self):

        # start clock
        start = time.time()

        # while not game finished
        while True:
            for player in [self.player1, self.player2]:



                # make the game go slower
                # time.sleep(1)

                # make the move for the players
                row, col = player.move()
                # change the baord
                self.board._add_move(row, col, player.name_player)
                # show the board
                self.board.drawboard()

                # check the status of the game
                status = self.game_status()

                # evaluate the speed at every move
                time_per_move = time.time() - start
                start = time.time()
                print("-------- time per move: {}".format(time_per_move))

                # if status is true the game has ended
                if status == True:
                    return None


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
        self.scores[self.m + self.n + (m+n-2) + (self.m - row) + col] += mark  # add +/-1 to the other diag index

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
        self.scores[self.m + self.n + (m+n-2) + (self.m - row) + col] -= mark  # add +/-1 to the other diag index

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


class PlayerAutomatic(Player):
    """
    Random player that playes automatically
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


class PlayerAI(Player):
    """
    Random player that playes automatically
    """


    def __init__(self, board, name_player, m, n, k):

        super().__init__(board, name_player, m, n, k)


    def move(self):

        # choose a position ranadomly
        row, col = self._select_move()


        print(row, col)

        return (row, col)


    def _select_move(self):

        # get the best move given the mini/max player
        if self.name_player == "X":
            best_move = self.max(0, -1e10, 1e10)[1]
        else:
            best_move = self.mini(0, -1e10, 1e10)[1]


        return best_move

    def max(self, depth, alpha, beta):

        # check if the depth is 0
        if self.board.board_status() != None:
            # print("Depth Reached: {}".format(depth))
            return self.board.board_status(), None

        # if starting with the max player
        max_eval = -1e10
        best_move = None

        # get the available moves
        next_moves = [(x, y) for x in range(self.m) for y in range(self.n) if self.board.is_valid(x, y)]

        # check all child nodes
        for move in next_moves:

            # make the move
            self.board._add_move(move[0], move[1], "X")

            eval = self.mini(depth - 1, alpha, beta)[0]

            # update alpha and check if it is bigger than beta
            alpha = max(alpha, eval)



            # check if it is bigger than previous
            if eval > max_eval:
                best_move = move
                max_eval = eval

            # undo move
            self.board._undo_move(move[0], move[1], "X")

            if alpha >= beta:
                break

        return max_eval, best_move



    def mini(self, depth, alpha, beta):

        # check if the depth is 0
        if self.board.board_status() != None:
            # print("Depth Reached: {}".format(depth))
            return self.board.board_status(), None

        min_eval = 1e10
        best_move = None

        # get the available moves
        next_moves = [(x, y) for x in range(self.m) for y in range(self.n) if self.board.is_valid(x, y)]

        # check all child nodes
        for move in next_moves:
            # make move
            self.board._add_move(move[0], move[1], "O")

            eval = self.max(depth - 1, alpha, beta)[0]

            # update beta and check if it is bigger than alpha
            beta = min(beta, eval)


            # check if it is bigger than previous
            if eval < min_eval:
                best_move = move
                min_eval = eval

            # undo move
            self.board._undo_move(move[0], move[1], "O")

            if alpha >= beta:
                break

        return min_eval, best_move





if __name__ == '__main__':

    # init define variables for the game
    m, n, k = 3, 3, 3
    # init class
    dummy_game = Game(m, n, k)
    # initialise game
    dummy_game.initialize_game()

    dummy_game.play()
