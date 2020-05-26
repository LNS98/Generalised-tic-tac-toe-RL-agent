"""
Class for RL player. 
"""

import numpy as np
import random
import torch
import collections

from players.base_player import Player 

# fix randomness
#random.seed(0)
#np.random.seed(0)
#torch.manual_seed(0)

class PlayerRL(Player):
    

    def __init__(self, board, name_player, m, n, k, hyper_parameters):

        super().__init__(board, name_player, m, n, k)

        # list of hyper-parameters for the agent 
        self.epsilon = hyper_parameters["epsilon"]
        self.gamma =  hyper_parameters["gamma"] 
        self.minimax_depth = hyper_parameters["minimax_depth"] # this is an input to the best_move (its 3 now)
        

        self.dqn = DQN(self.gamma, m, n)
        
        # path to where to save the weights
        self.weights_dir = f"./weights/{m}_{n}_{k}_{self.name_player}/"
        self.weights_file = f"{self.epsilon}_{self.gamma}_{self.minimax_depth}.pth"

        # check if weightd for this network already exist
        try:
            self.dqn.q_network.load_state_dict(torch.load(self.weight_path)) 
        except:
            print("{} never trained before".format(self.name_player))

        self.buffer = ReplayBuffer()



    def move(self):

        # choose a position ranadomly
        row, col = self._select_move()

        return (row, col)


    def _select_move(self):

        # get a random move
        random_move = random.choice([(x, y) for x in range(self.m) for y in range(self.n) if self.board.is_valid(x, y)])

        # get the best move given the mini/max player
        if self.name_player == "X":
            best_move = self.max(self.minimax_depth, -1e10, 1e10)[1]
        else:
            best_move = self.mini(self.minimax_depth, -1e10, 1e10)[1]  

        # select with prob epsilon the random move
        moves = [best_move, random_move]
        choice_index = np.random.choice(len(moves), 1, replace=True, p=[1-self.epsilon, self.epsilon])[0]

        selected_move = moves[choice_index]

        return selected_move

    def max(self, depth, alpha, beta):

        # check if the depth is 0
        if self.board.board_status() != None or depth < 0:

            # return the value at this position
            batch_input_tensor = torch.tensor(self.board.scores).float()
            value = self.dqn.q_network.forward(batch_input_tensor)

            # print("Depth Reached: {}".format(depth))
            return value, None

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
        if self.board.board_status() != None or depth < 0:
            # return the value at this position
            batch_input_tensor = torch.tensor(self.board.scores).float()
            value = self.dqn.q_network.forward(batch_input_tensor)

            # print("Depth Reached: {}".format(depth))
            return value, None


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


# The Network class inherits the torch.nn.Module class, which represents a neural network.
class Network(torch.nn.Module):
    
    # THIS NETWORK LOOKS WAY TO HUGE 
    def __init__(self, input_dimension, output_dimension):
    
        super(Network, self).__init__()
      
        self.layer_1 = torch.nn.Linear(in_features=input_dimension, out_features=50)
        self.layer_2 = torch.nn.Linear(in_features=50, out_features=50)
        self.output_layer = torch.nn.Linear(in_features=50, out_features=output_dimension)


    def forward(self, input):
        layer_1_output = torch.nn.functional.relu(self.layer_1(input))
        layer_2_output = torch.nn.functional.relu(self.layer_2(layer_1_output))
        output = self.output_layer(layer_2_output)
        return output

# The DQN class determines how to train the above neural network.
class DQN:

    # The class initialisation function.
    def __init__(self, gamma, m, n):
        # Create a Q-network, which predicts the q-value for a particular state.
        self.q_network = Network(input_dimension=m+n+2*(m+n-1), output_dimension=1)

        # create target network
        # self.q_target_network = Network(input_dimension=m+n+2*(m+n-1), output_dimension=1)
        # place the weights of the q_network to the target
        # self.q_target_network.load_state_dict(self.q_network.state_dict())

        # Define the optimiser which is used when updating the Q-network. The learning rate determines how big each gradient step is during backpropagation.
        self.optimiser = torch.optim.Adam(self.q_network.parameters(), lr=0.001)
        self.gamma = gamma

    # Function that is called whenever we want to train the Q-network. Each call to this function takes in a transition tuple containing the data we use to update the Q-network.
    def train_q_network(self, transition):
        # Set all the gradients stored in the optimiser to zero.
        self.optimiser.zero_grad()
        # Calculate the loss for this transition.
        loss = self._calculate_loss(transition)
        # Compute the gradients based on this loss, i.e. the gradients of the loss with respect to the Q-network parameters.
        loss.backward()
        # Take one gradient step to update the Q-network.
        self.optimiser.step()
        # Return the loss as a scalar

        return loss.item()

    # Function to calculate the loss for a particular transition.
    def _calculate_loss(self, transitions):

        batch_input_tensor = torch.tensor([transition[0] for transition in transitions]).float()
        batch_labels_tensor = torch.tensor([transition[1] for transition in transitions]).float().view(5, 1)

        # Do a forward pass of the network using the inputs batch
        network_prediction = self.q_network.forward(batch_input_tensor)

        # Compute the loss based on the label's batch
        loss = torch.nn.MSELoss()(network_prediction, batch_labels_tensor)


        return loss



class ReplayBuffer:

    def __init__(self):
        # initilaise an empty deque
        self.container = collections.deque(maxlen=1_000_000)


    def add_transition(self, transition):
        # append the transition to the deque
        self.container.append(transition)


    def random_batch(self, batch_number):
        """
        Generate a random batch of transitions from the ReplayBuffer
        """

        # generate a bunch of random indicies in the correct length
        batch_indices = np.random.choice(range(len(self.container)), batch_number, replace = False)

        batch = [self.container[batch_index] for batch_index in batch_indices]

        return batch
