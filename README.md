# RL agent for generalised tic-tac-toe (m,n,k-game)

## The m,n,k-game (tic-tac-toe)

[Wikipedia](https://en.wikipedia.org/wiki/M,n,k-game) description: 
'An m,n,k-game is an abstract board game in which two players take turns in placing a stone of their color on an m×n board, the winner being the player who first gets k stones of their own color in a row, horizontally, vertically, or diagonally. Thus, tic-tac-toe is the 3,3,3-game and free-style gomoku is the 15,15,5-game. An m,n,k-game is also called a k-in-a-row game on an m×n board.'                 

## Description 

The repository contains an implementation to play an [m,n,k-game](https://en.wikipedia.org/wiki/M,n,k-game). Two AI agents are included: a minimiax algorithm with alpha-beta pruning and a Reinforcment Learing (RL) agent (for more detail on the implementation look at `/playres/rl_agent.py`). 
No heuristics are included for the minimax agent and as such it always searches until a winner is found in the game which makes it too slow for any value of m, n, k > 3. This is the reason for the introduction of the RL agent: rather than introduce human-designed huristics, let the agent reach human level (or better) play through RL and self-play.


## Usage


- Clone the repository to the desired location `git clone https://github.com/LNS98/Generalised-tic-tac-toe-RL-agent.git`

- Install the dependencies in the `requirements.txt` file. 

- To play a game run `python play.py` followed by the following command line arguments:
    - m: a positive integer defining the number of rows in the game
    - n: a positive integer defining the number of columns in the game
    - k: number of consecutives objects for a win 
    - player X type: Choose between human, a minimax algorithm (with alpha-beta pruning) and a RL player 
    - Player O: Choose between human, a minimax algorithm (with alpha-beta pruning) and a RL player

<strong> Note: Minimax will be to slow to play against for values of m > 3, n > 3, k >3, unless you are really (really) patient as it will search until a winning position is found (in other words, no maximum depth can be set). It is the reason for introducing the RL agent in the first place. </strong>


- To train a specific RL agent run `python train_rl_agent.py` followed by the following command line arguments:
    - m: a positive integer defining the number of rows in the game
    - n: a positive integer defining the number of columns in the game
    - k: number of consecutives objects for a win 
  

<strong>Note: All the hyper-parameters for the agents are set as constants at the top of this file. </strong>
 
## Repository Guide 

- `play.py` - Used to play the game, need to specify arguments as specified in usage. 

- `train_rl_agent.py` - Used to train the RL agent, hyper-parameters for training have to be specified.

- `players/` - Contains files for each of the implementation of the agents.
           
     - `base_player.py` - Contains class for the human player and the parent class used in the other two agent class.
     - `minimaxAlphaBeta_player.py` - Contains class for the minimax (with alpha-beta pruning) agent implementation.
     - `rl_player.py` - Contains class for the RL agent implementation.
            
- `setup/` - Contains files for the implentation of the board and game.
           
     - `board.py` - Contains class for the board used in the creation of the game.
     - `game.py` - Contains class for the implementation of the game.  

- `weights/` - Folder used to store the weights of the RL algorithms that are trained. Files are created for a specific game selected to which each player's weights will be saved. For example if weights are selected to be saved for a m=3, n=3, k=3 game then both player X and O will be saved in folders: `weights/3_3_3_x.pth` and `weights/3_3_3_o.pth`
