import numpy as np
import torch

import Nets
from eval import pit_against_network, pit_human, hist_compare
from Connect4Game import Connect4Game

if __name__ == '__main__':

    # initialize networks
    net = Nets.Dense()
    net.load_state_dict(torch.load('dense/dense9.pth'))
    net.eval()
    aux_net = Nets.Dense()

    # initialize game
    game = Connect4Game()


    wins,draws,losses = pit_against_network(net,aux_net,game,100)

    print('Wins:',wins)
    print('Losses:',losses)
    print('Draws:',draws)

    #pit_human(net,game)
    #hist_compare(game,name='dense',num_games=100)
