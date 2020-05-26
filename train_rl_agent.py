"""
File containing scripts used to train the RL players. 
"""

import argparse
import torch
import os  

from setup.board import Board
from setup.game import Game
from players.rl_player import PlayerRL

def parseArguments():

    # create the parser 
    pars = argparse.ArgumentParser()

    # specify the type of game to be player (these will have to be restricted in the future) 
    pars.add_argument("m", help="value of 'm', the number of rows in the game", default=3) 
    pars.add_argument("n", help="value of 'n', the number of coloumns  in the game", default=3) 
    pars.add_argument("k", help="value of 'k', the number of consecutives objects for a win ", default=3) 
    # specify the parameters for training 
    pars.add_argument("starting_epsilon", help="exploration rate at the start of training", default=1)
    pars.add_argument("gamma", help="discount factor to reducre future rewards", default=1)
    pars.add_argument("minimax_depth", help="depth to which minimax search is conducted", default=3)


    args = pars.parse_args()

    return args


def train(m, n, k, hyper_pars,  n_games=1_000):
   
    # value to decrease the exploration
    delta = 0.02

    # init class
    dummy_game = Game(m, n, k, display=False)
    
    # init the two agents 
    player1 = PlayerRL(dummy_game.board, "X", m, n, k, hyper_pars)
    player2 = PlayerRL(dummy_game.board, "O", m, n, k, hyper_pars)

    # keep track of the wins 
    wins_1 = 0
    wins_2 = 0
    ties = 0

    print("--------------------- Start Trainning Games ------------------------------") 
    # play n_games
    for i in range(n_games):
        # initialise game
        dummy_game.initialize_game(player1, player2)

        episode_states, Gt = dummy_game.play()

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
                # check if the directory exists 
                if not os.path.exists(player1.weights_dir):
                    os.mkdir(player1.weights_dir)
                print("Save weights for player 1 and reset exploration")
                player1.epsilon = 1
                torch.save(player1.dqn.q_network.state_dict(), player1.weights_dir + player1.weights_file)
      
            if isinstance(player2, PlayerRL):
                if not os.path.exists(player2.weights_dir):
                    os.mkdir(player2.weights_dir)
                print("Save weights for player 2 and reset exploration")
                player2.epsilon = 1
                torch.save(player2.dqn.q_network.state_dict(), player2.weights_dir + player2.weights_file)


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




# Main entry point
if __name__ == "__main__":

    args = parseArguments()


    # print the args
    for arg in args.__dict__:
        print(f"{arg}: {args.__dict__[arg]}")
    
    
    # init define variables for the game
    m, n, k = int(args.m), int(args.n), int(args.k)
    
    # define the hyper-paramets 
    hyper_pars = {"epsilon": int(args.starting_epsilon), "gamma": int(args.gamma), "minimax_depth": int(args.minimax_depth)}

    # run the training 
    train(m, n, k, hyper_pars)





