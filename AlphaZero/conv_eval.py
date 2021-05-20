import numpy as np
import torch

import Nets
from eval import pit_against_network, pit_human, hist_compare, pit_onelookahead
from Connect4Game import Connect4Game

if __name__ == '__main__':

    # initialize networks
    net = Nets.Conv()
    net.load_state_dict(torch.load('convbkup/conv10.pth'))
    net.eval()
    aux_net = Nets.Conv()
    #aux_net.load_state_dict(torch.load('convbkup/conv1.pth'))
    aux_net.eval()

    # initialize game
    game = Connect4Game()


    """ wins,draws,losses = pit_against_network(aux_net,net,game,100)

    print('Wins:',wins)
    print('Losses:',losses)
    print('Draws:',draws) """
   
    #pit_human(aux_net,game)
    hist_compare(game,name='conv',num_games=100)
    """ wins,draws,losses = pit_onelookahead(game,aux_net,1)

    print('Wins:',wins)
    print('Losses:',losses)
    print('Draws:',draws)  """
