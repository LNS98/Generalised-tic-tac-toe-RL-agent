"""
Board class used in the m, n, k game.
"""



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

    # THIS IS NOT USED AND AT THE FIRST ROUND IT DOESNT CHECK FOR VALIDITY 
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


