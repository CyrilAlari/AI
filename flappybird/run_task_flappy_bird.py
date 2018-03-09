#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Guide : Installation...
 http://pygame-learning-environment.readthedocs.io/en/latest/user/home.html
"""

from ple.games.flappybird import FlappyBird
from ple import PLE
from evolver_flappy_bird import QLearnerEvolverFlappy
import numpy as np
import time
import math
from collections import deque
from random import randint

def process_state(state):
    #print(state)
    return np.array([list(state.values())])

#launch the environment
game = FlappyBird()
p = PLE(game, fps=30, display_screen=True, force_fps=False, state_preprocessor=process_state)
p.init()
game.ple = p

#number of the game we launch. This way we avoid overwriting files
#that were created with other params
number_experiment = randint(0,10000000)

#agent
action_set = p.getActionSet()
agent = QLearnerEvolverFlappy(len(action_set),p.getGameStateDims()[1])
agent.should_epsilon_decay = False #to control the decay differently
# agent.load("flappy1_100.h5")

nb_games = 1 #game counter
nb_frames = 0 #frame counter
score_game = 0 #score of the current game

#to average last losses and scores
last_losses = deque(maxlen=1000)
last_500_games_score = deque(maxlen=500)

#flags to write in files only once per x game
flag_game_10 = False
flag_game_100 = False
flag_game_500 = False

#Epsilon constants
EXPLORE = 5000000 #small is 300000, big is 5000000
FINAL_EPSILON = 0.0001
INITIAL_EPSILON = 0.1



while 1:

    nb_frames += 1

    if p.game_over():
        p.reset_game()
        nb_games += 1
        print("score for this game :", score_game)
        last_500_games_score.append(score_game)
        score_game = 0

    #classic part for action in a RL game
    observation = p.getGameState()
    action = agent.pickAction(observation)
    reward = p.act(p.getActionSet()[action])
    agent.remember(observation, action, reward, p.getGameState(), p.game_over())

    #Epsilon decaying
    if nb_frames<EXPLORE:
        agent.epsilon = (INITIAL_EPSILON*(EXPLORE-nb_frames)+FINAL_EPSILON*nb_frames)/EXPLORE
    else:
        agent.epsilon = FINAL_EPSILON

    #when the bird manage to avoid a pipe
    if reward>0.5:
        score_game += 1

    #we don't train at the beginning, only take random actions to fill memory
    if nb_frames==agent.memory.maxlen:
        print("starting training")

    #update the fast NN
    if nb_frames%10==0 and nb_frames>=agent.memory.maxlen:
        agent.updateEvolver()
        # for layer in agent.model.layers:
        #     weights = layer.get_weights()
        #     print(weights)
        #     break
        last_losses.append(round(sum(abs((-agent.model.predict(observation)+reward+agent.gamma*agent.model.predict(p.getGameState()))[0])),3))

    #update the slow NN
    if nb_frames%300==0 and nb_frames>=len(agent.memory):
        agent.updateModel()

    #saving the model
    if nb_frames%5000==0:
        print("5000 frames, saving model")
        print("nb frames since beginning :", nb_frames)
        agent.save("weights_nn/flappy_new_features_100_"+str(number_experiment)+".h5")


    #writing scores in files
    if nb_games%500==0 and not flag_game_500:
        with open('results/file_score_nf_'+str(number_experiment)+'.txt', 'a') as the_file:
            the_file.write('\n'+str(sum(last_500_games_score)/500))
        flag_game_500 = True
    if nb_games%500==1:
        flag_game_500 = False

    #writing loss in files
    if nb_games%100==0 and not flag_game_100:
        with open('results/file_loss_nf_'+str(number_experiment)+'.txt', 'a') as the_file:
            the_file.write('\n'+str(sum(np.array(last_losses)[-100:])/100))
        flag_game_100 = True
    if nb_games%100==1:
        flag_game_100 = False

    #printing some info
    if nb_games%10==0 and not flag_game_10:
        print(nb_games)
        #print("j=",j,"/ Score : ",p.score()," / Success : ",catch," / Fail : ",fail, " / Rapport : ", 0 if fail==0 else round(catch/fail,3))
        print("loss :", "inf" if len(last_losses)==0 else sum(last_losses)/len(last_losses))
        print("epsilon :", agent.epsilon)
        print("\n")
        flag_game_10 = True
    if nb_games%10==1:
        flag_game_10 = False
