from typing import Deque
import numpy as np
from numpy.core import function_base
import torch
import torch.nn as nn
import torch.nn.functional as functions
import matplotlib.pyplot as plt
from collections import deque
import random

from Connect4Game import Connect4Game
from Nets import Conv
from Nets import Alphaloss
from MCTS import MCTS


## Episode Generator
def gen_episode(game,net):

    examples = []
    s = game.getInitBoard()
    player = 1  #initially
    mcts = MCTS(game,net)
    episodeStep = 0

    while True:
        episodeStep += 1
        canonicalBoard = s*player   #the board from the current players perspective
        temp = int(episodeStep<15)  #the search should return deterministic results after some steps
        pi = mcts.getActionProb(canonicalBoard,temp=temp)
        for b,p in game.getSymmetries(canonicalBoard,pi):
            examples.append([b,player,p,None])
        action = np.random.choice(len(pi),p=pi)
        s, player = game.getNextState(s,player,action)

        r = game.getGameEnded(s,player)

        if(r!=0):
            return [(x[0], x[2], r * ((-1) ** (x[1] != player))) for x in examples]


## Train function
def train(net,optimizer,samples):
    losses = []

    s,pi,z = zip(*samples)
    s = torch.from_numpy(np.array(s)).float()
    pi = torch.from_numpy(np.array(pi)).float()
    z = torch.from_numpy(np.array(z)).float()
    z = z.view(-1,1)

    optimizer.zero_grad()
    P,v = net(s)
    alphaloss = Alphaloss()
    loss = alphaloss(z,v,pi,P)
    loss.backward()
    losses.append(loss.item())
    optimizer.step()

    return losses


## Train Loop
def learn(game,net,num_iter,reps=10,continuous_graph=False):

    optimizer = torch.optim.Adam(net.parameters(),lr=0.0005)
    net.load_state_dict(torch.load('conv/conv10.pth'))
    reps = 10
    samples = deque(maxlen=400)
    losses = []
    sample_size = 64

    for i in range(1,num_iter+1):
        samples.clear()
        print(i,end='\r')
        for eps in range(5):
            samples += gen_episode(game,net)
        print('Samples Created: ',len(samples))
        for j in range(reps):
            try:
                losses += train(net,optimizer,random.sample(samples,sample_size))
            except:
                print('Sample Size too small')
                losses += train(net,optimizer,random.sample(samples,len(samples)))
        if(continuous_graph):
            plt.plot(losses)
            plt.savefig('conv1loss.png')
        if(i%10==0):
            torch.save(net.state_dict(),'conv/conv'+str(int(i/10))+'.pth')
        
    #torch.save(net.state_dict(),'dense1.pth')
    plt.plot(losses)
    plt.savefig('conv1loss.png')

if __name__ == '__main__':

    game = Connect4Game()
    net = Conv()
    learn(game,net,100,continuous_graph=True)