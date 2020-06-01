"""
File containing scripts used to train the RL players. 
"""

import argparse
import torch
import matplotlib.pyplot as plt 


from setup.board import Board
from setup.game import Game
from players.rl_player import PlayerRL

# add all constants
SAVE_MODEL = True
DEBUG = False
MINIBATCH_SIZE = 16
MIN_REPLAY_SIZE = 5 + MINIBATCH_SIZE 

GAMMA = 1
EPSILON = 1
EPSILON_DECAY = 0.05
MINIMAX_DEPTH = 3
MIN_EPSILON = 0.001

DISPLAY_STATS_EVERY = 100
SAVE_MODEL_EVERY = 400
EPISODES = 1000

def parseArguments():

    # create the parser 
    pars = argparse.ArgumentParser()

    # specify the type of game to be player (these will have to be restricted in the future) 
    pars.add_argument("m", help="value of 'm', the number of rows in the game", default=3) 
    pars.add_argument("n", help="value of 'n', the number of coloumns  in the game", default=3) 
    pars.add_argument("k", help="value of 'k', the number of consecutives objects for a win ", default=3) 

    args = pars.parse_args()

    return args


def train(m, n, k, hyper_pars):
    
    # init class
    dummy_game = Game(m, n, k, display=False)
   
    # define the hyper-parameters for the agent 

    # init the two agents 
    player1 = PlayerRL(dummy_game.board, "X", m, n, k, hyper_pars)
    player2 = PlayerRL(dummy_game.board, "O", m, n, k, hyper_pars)
    
    metrics = {"loss_X": [], "loss_O": []}

    print("--------------------- Start Trainning Games ------------------------------") 
    # play n_games
    for i in range(EPISODES):
        # initialise game
        dummy_game.initialize_game(player1, player2)

        episode_states, Gt = dummy_game.play()
        
        # process for each player 
        for player in [player1, player2]:
            # decrease the explortion rate 
            player.epsilon = max(player.epsilon-EPSILON_DECAY, MIN_EPSILON)
            
            # after every 100 games store the weights
            if SAVE_MODEL == True and i % SAVE_MODEL_EVERY == 0:
                print("Weights saved")
                torch.save(player.dqn.q_network.state_dict(), player.weights_path) 
             # add the transition to the ReplayBuffer
            for state in episode_states:
                player.buffer.add_transition((state, Gt))

            # train network
            if len(player.buffer.container) > MIN_REPLAY_SIZE:
                batch = player.buffer.random_batch(MINIBATCH_SIZE)
                # add the training by computing the loss at each step
                loss = player.dqn.train_q_network(batch)
                metrics[f"loss_{player.name_player}"].append(loss)
                            
                    
        if DEBUG==True and i % DISPLAY_STATS_EVERY == 0:
            plt.plot([i for i in range(len(metrics["loss_X"]))], metrics["loss_X"])
            plt.plot([i for i in range(len(metrics["loss_O"]))], metrics["loss_O"])
            plt.show()


# Main entry point
if __name__ == "__main__":
    

    # fix randomness
    #random.seed(0)
    #np.random.seed(0)
    #torch.manual_seed(0)

    args = parseArguments()

    # print the args
    for arg in args.__dict__:
        print(f"{arg}: {args.__dict__[arg]}")
    
    # init define variables for the game
    m, n, k = int(args.m), int(args.n), int(args.k)
    
    # define the hyper-paramets 
    hyper_pars = {"epsilon": EPSILON, "gamma": GAMMA, "minimax_depth": MINIMAX_DEPTH}

    # run the training 
    train(m, n, k, hyper_pars)


