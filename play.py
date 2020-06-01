"""
Script that allows you to play the m,n,k game with pre-defined setting specified by the user.
"""


import argparse

from setup.game import Game 
from players.base_player import Player
from players.rl_player import PlayerRL
from players.minimaxAlphaBeta_player import PlayerMiniMax


def parseArguments():

    # create the parser
    pars = argparse.ArgumentParser()


    # specify the type of game to be player (these will have to be restricted in the future) 
    pars.add_argument("m", help="value of 'm', the number of rows in the game", default=3) 
    pars.add_argument("n", help="value of 'n', the number of coloumns  in the game", default=3) 
    pars.add_argument("k", help="value of 'k', the number of consecutives objects for a win ", default=3) 
    # specify the two players to play with
    pars.add_argument("player_1", help="choose between human, RL, or minimax players", choices=["human","rl","minimax"], default="human") 
    pars.add_argument("player_2", help="choose between human, RL, or minimax players", choices=["human","rl","minimax"], default="human") 

    args = pars.parse_args()

    return args


if __name__ == "__main__":
    
    args = parseArguments()

    # print the args
    for arg in args.__dict__:
        print(f"{arg}: {args.__dict__[arg]}")

    m, n, k = int(args.m), int(args.n), int(args.k)

    # init the game obj
    game = Game(m, n, k)
    
    # specify the players
    player_dict = {"human":  Player, "rl": PlayerRL, "minimax": PlayerMiniMax}
    
    # use default hyper-par for rl agent
    if args.player_1 == "rl":
        hyper_pars = {"epsilon": 1, "gamma": 1, "minimax_depth": 3} 
        player_x = player_dict[args.player_1](game.board, "X", m, n, k, hyper_pars, mode="infer") 
    else:
        player_x = player_dict[args.player_1](game.board, "X", m, n, k)
   
   # same as for player 1
    if args.player_2 == "rl":
        hyper_pars = {"epsilon": 1, "gamma": 1, "minimax_depth": 3} 
        player_o = player_dict[args.player_2](game.board, "O", m, n, k, hyper_pars, mode="infer")
    else:
        player_o = player_dict[args.player_2](game.board, "O", m, n, k)

    wins_1 = 0
    wins_2 = 0
    ties = 0



    print(
        "                       **********************************************************                                    ")
    print(
        "                                    Welcome to the m, n, k game!                                                     ")
    print(
       f"                                    Game settings: m={m}, n={n}, k={k}                                               ")
    print(
        "                       **********************************************************                                    ")
            

    while True:
       # initialise game
       game.initialize_game(player_x, player_o)
       

       episode_states, Gt = game.play()

       # print the numebr of wins by each player
       if Gt == 1:
           wins_1 += 1
       elif Gt == -1:
           wins_2 += 1
       else:
           ties += 1
      
       # ask the user if he wants to play another game
       again = input("Would you like to play another game? (y/n)")
       if again == "n":
           print("Thank you for playing! See you next time")
           break
           
       print("Player 1: {}, Player 2: {}, Ties: {}".format(wins_1, wins_2, ties))


