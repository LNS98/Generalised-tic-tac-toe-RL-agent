
from mnk_game_rl_agent import *
from player import *


if __name__ == "__main__":
    # init define variables for the game
    m, n, k = 3, 3, 3

    # init class
    dummy_game = Game(m, n, k)

    player1 = Player(dummy_game.board, "X", m, n, k)
    player2 = PlayerRL(dummy_game.board, "O", m, n, k, 0)


    wins_1 = 0
    wins_2 = 0
    ties = 0


    while True:
        # initialise game
        dummy_game.initialize_game(player1, player2)

        episode_states, Gt = dummy_game.play(debug=True)

        # print the numebr of wins by each player
        if Gt == 1:
            wins_1 += 1
        elif Gt == -1:
            wins_2 += 1
        else:
            ties += 1
        print("Player 1: {}, Player 2: {}, Ties: {}".format(wins_1, wins_2, ties))
