## This File is for Directly Playing against 'alphanet.pth'
# To play against any other network, change the network name below
import numpy as np
import torch
from eval import pit_human
from Connect4Game import Connect4Game
from Nets import Conv

if __name__ == '__main__':

    #initialize network
    net = Conv()
    net.load_state_dict(torch.load('alphanet.pth'))

    #initialize game
    game = Connect4Game()

    pit_human(net,game)
