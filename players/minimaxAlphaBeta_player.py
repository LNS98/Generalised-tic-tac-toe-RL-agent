"""
AI player class running on a minimax algorithm with alpha-beta pruning. 
"""

from players.base_player import Player

class PlayerMiniMax(Player):
    """
    AI player that playes using the minimax algorithm with alpha-beta pruning
    """


    def __init__(self, board, name_player, m, n, k):

        super().__init__(board, name_player, m, n, k)


    def move(self):
        
        # choose a position ranadomly
        row, col = self._select_move()
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

