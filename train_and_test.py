
from mnk_game_rl_agent import *
from player import *

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)


# Main entry point
if __name__ == "__main__":

        # init define variables for the game
        m, n, k = 4, 4, 4

        # value to decrease the exploration
        delta = 0.02

        # init class
        dummy_game = Game(m, n, k)

        player1 = PlayerRL(dummy_game.board, "X", m, n, k, 1)
        player2 = PlayerRL(dummy_game.board, "O", m, n, k, 1)


        wins_1 = 0
        wins_2 = 0
        ties = 0

        for i in range(1_000_000):
            # initialise game
            dummy_game.initialize_game(player1, player2)

            episode_states, Gt = dummy_game.play(debug=False)

            # print the numebr of wins by each player
            if Gt == 1:
                wins_1 += 1
            elif Gt == -1:
                wins_2 += 1
            else:
                ties += 1
            print("Player 1: {}, Player 2: {}, Ties: {}".format(wins_1, wins_2, ties))


            # after every 100 games store the weights
            if i % 100 == 0:

                if isinstance(player1, PlayerRL):
                    print("Save weights for player 1 and reset exploration")
                    player1.epsilon = 1
                    torch.save(player1.dqn.q_network.state_dict(), "{}_{}_{}_{}.pth".format(m, n, k, player1.name_player))
                if isinstance(player2, PlayerRL):
                    print("Save weights for player 2 and reset exploration")
                    player2.epsilon = 1
                    torch.save(player2.dqn.q_network.state_dict(), "{}_{}_{}_{}.pth".format(m, n, k, player2.name_player))


            # add the transition to the ReplayBuffer
            for state in episode_states:
                    if isinstance(player1, PlayerRL):
                        if player1.epsilon - delta > 0:
                            player1.epsilon -= delta
                        player1.buffer.add_transition((state, Gt))
                    if isinstance(player2, PlayerRL):
                        if player2.epsilon - delta > 0:
                            player2.epsilon -= delta
                        player2.buffer.add_transition((state, Gt))


            # train network
            if isinstance(player1, PlayerRL):
                if len(player1.buffer.container) > 5:
                        batch = player1.buffer.random_batch(5)
                        # add the training by computing the loss at each step
                        loss = player1.dqn.train_q_network(batch)

            if isinstance(player2, PlayerRL):
                if len(player2.buffer.container) > 5:
                    batch = player2.buffer.random_batch(5)
                    # add the training by computing the loss at each step
                    loss = player2.dqn.train_q_network(batch)
