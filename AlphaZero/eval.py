## This File contains various utilities for evaluating the trained networks


import numpy as np
import random
from MCTS import MCTS
import torch
import Nets
import matplotlib.pyplot as plt


def pit_against_network(player1,player2,game,num_games,render=False):

    wins = 0
    draws = 0

    print('Pitting Networks...')
    for game_no in range(num_games):
        player = 1
        print(game_no,end='\r')
        s = game.getInitBoard()
        mcts_player1 = MCTS(game,player1)
        mcts_player2 = MCTS(game,player2)
        while True:
            if render:
                print_board_pretty(s)
            #player1 move
            #temp=0 ensures deterministic moves
            pi = mcts_player1.getActionProb(s*player,temp=0)   #s*player is the canonical board
            a = np.random.choice(len(pi),p=pi)
            new_state = game.getNextState(s,player,a)[0]
            if game.getGameEnded(new_state,player) != 0:
                reward = game.getGameEnded(new_state,player)
                if(reward == 1e-4):
                    if render:
                        print_board_pretty(new_state)
                    draws += 1
                elif(reward == 1):
                    if render:
                        print_board_pretty(new_state)
                    wins += 1
                break
            s = new_state
            player *= -1

            #player2 move
            pi = mcts_player2.getActionProb(s*player,temp=0)
            a = np.random.choice(len(pi),p=pi)
            new_state = game.getNextState(s,player,a)[0]
            if game.getGameEnded(new_state,player) != 0:
                reward = game.getGameEnded(new_state,player)
                if(reward == 1e-4):
                    if render:
                        print_board_pretty(new_state)
                    draws += 1
                break
            s = new_state
            player *= -1

    return wins, draws, num_games-wins-draws   

def print_board_pretty(board):      #utility function for pit_human
    print('|-----------------|')
    print('|  1 2 3 4 5 6 7  |')
    shape = board.shape
    for i in range(shape[0]):
        print('|',end='  ')
        for j in range(shape[1]):
            if(board[i,j] == 0):
                print('_',end=' ')
            if(board[i,j] == 1):
                print('X',end=' ')
            if(board[i,j] == -1):
                print('O',end=' ')
        print(' |')

    print('|_________________|')
    print()

def pit_human(net,game):
    
    print(net.eval())
    
    print('Starting Game')
    player = 1
    s = game.getInitBoard()
    mcts = MCTS(game,net)
    while True:
        #AI turn
        pi = mcts.getActionProb(s*player,temp=0)
        a = np.random.choice(len(pi),p=pi)
        new_state = game.getNextState(s,player,a)[0]
        if game.getGameEnded(new_state,player) != 0:
            reward = game.getGameEnded(new_state,player)
            if(reward == 1e-4):
                print_board_pretty(new_state)
                print('DRAW')
            elif(reward == 1):
                print_board_pretty(new_state)
                print('AI Wins')
            break
        s = new_state
        player *= -1
        print_board_pretty(s)

        #Human Turn
        a = int(input('Enter action [1-7]...')) - 1
        while not(a<7 and a>=0):
            a = int(input('Enter Valid Action [1-7]...'))
        new_state = game.getNextState(s,player,a)[0]
        if game.getGameEnded(new_state,player) != 0:
            reward = game.getGameEnded(new_state,player)
            if(reward == 1e-4):
                print_board_pretty(new_state)
                print('Draw')
            elif(reward == 1):
                print_board_pretty(new_state)
                print('Human Wins')
            break
        s = new_state
        player *= -1
        


def pit_onelookahead(game,net,num_games):
    
    wins = 0
    draws = 0
    print(net.eval())
    print('Pitting Against 1 Step Lookahead Player')
    player = 1
    s = game.getInitBoard()
    mcts = MCTS(game,net)
    for game_no in range(num_games):
        print(game_no,'\r')
        while True:
            #AZ turn
            print_board_pretty(s)
            #pi = mcts.getActionProb(s*player,temp=0)
            #a = np.random.choice(len(pi),p=pi)
            a = int(input('Enter Action')) - 1
            new_state = game.getNextState(s,player,a)[0]
            if game.getGameEnded(new_state,player) != 0:
                reward = game.getGameEnded(new_state,player)
                if(reward == 1e-4):
                    print_board_pretty(new_state)
                    draws += 1
                if(reward == 1):
                    print_board_pretty(new_state)
                    wins += 1
                break
            s = new_state
            player *= -1

            #lookahead turn
            valid_actions = []
            best_action = -1
            for a_i,valid in enumerate(game.getValidMoves(s,player)):
                if valid:
                    valid_actions.append(a_i)
                    reward = game.getGameEnded(game.getNextState(s,player,a_i)[0],player)
                    if reward == 1:
                        best_action = a_i
                        break
            if best_action == -1:
                best_action = random.choice(valid_actions)

            new_state = game.getNextState(s,player,best_action)[0]
            if game.getGameEnded(new_state,player) != 0:
                reward = game.getGameEnded(new_state,player)
                if(reward == 1e-4):
                    print_board_pretty(new_state)
                    draws += 1
                if(reward == 1):
                    print_board_pretty(new_state)
                    wins += 1
                break
            s = new_state
            player *= -1
    
    return wins,draws,num_games-wins
            


# Compare network to its trained history
def hist_compare(game,name='dense',num_games=100):      #currently string default is 'dense'

    wins_list = []
    losses_list = []
    draws_list = []
    for net_no in range(1,11):
        if net_no == 1:
            player2 = Nets.Conv()
            player2.eval()                                                              # dense or conv
        else:
            player2 = Nets.Conv()                                                              # dense or conv
            player2.load_state_dict(torch.load(name+'/'+name+str(int(net_no-1))+'.pth'))        # dense or conv
            player2.eval()
        player1 = Nets.Conv()                                                                  # dense or conv
        player1.load_state_dict(torch.load(name+'/'+name+str(int(net_no))+'.pth'))              # dense or conv
        player1.eval()
        print('Pitting',net_no-1,'and',net_no)
        wins,draws,losses = pit_against_network(player1,player2,game,num_games)
        print('wins:',wins)
        print('draws:',draws)
        print('losses:',losses)
        wins_list.append(wins)
        draws_list.append(draws)
        losses_list.append(losses)
    
    plt.plot(wins_list,color='green')
    plt.plot(losses_list,color='red',alpha=0.3)
    plt.savefig('HistComp'+name+'.png')

# Slightly Different Function to compare all networks to the best till then
def hist_compare_best(game,name='dense',num_games=100):

    wins_list = []
    losses_list = []
    draws_list = []
    best_wins = 0
    best_net = 0
    for net_no in range(1,11):
        if best_net == 0:
            player2 = Nets.Dense()                                                             # dense or conv
        else:
            player2 = Nets.Dense()                                                             # dense or conv
            player2.load_state_dict(torch.load(name+'/'+name+str(int(best_net))+'.pth'))       # dense or conv
            player2.eval()
        player1 = Nets.Dense()                                                                 # dense or conv
        player1.load_state_dict(torch.load(name+'/'+name+str(int(net_no))+'.pth'))             # dense or conv
        player1.eval()
        print('Pitting',net_no-1,'and',net_no)
        wins,draws,losses = pit_against_network(player1,player2,game,num_games)
        print('wins:',wins)
        print('draws:',draws)
        print('losses:',losses)
        if(wins>=best_wins):
            best_wins = wins
            best_net = net_no
        print('Best Net:',best_net)
        wins_list.append(wins)
        draws_list.append(draws)
        losses_list.append(losses)
    
    plt.plot(wins_list,color='green')
    plt.plot(losses_list,color='red',alpha=0.3)
    plt.savefig('BestHistComp'+name+'.png')
