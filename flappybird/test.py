from ple.games.waterworld import WaterWorld
from ple import PLE
from evolver2 import QLearnerEvolver
import numpy as np
import time
import math
from collections import deque

import random
import numpy as np
from collections import deque
from keras.models import Sequential, clone_model, load_model
from keras.layers import Dense
from keras.optimizers import Adam


# agent = QLearnerEvolver(len(action_set), p.getGameStateDims())
# agent.load("model4_48_48.h5")

model = Sequential()
model.add(Dense(48, input_dim=66, activation='relu'))
model.add(Dense(48, activation='relu'))
model.add(Dense(5, activation='linear'))
model.compile(loss='mse',
              optimizer=Adam(lr=0.05))

model.load_weights("model3_48_48.h5")

for layer in model.layers:
    weights = layer.get_weights()
    print(weights)
    break
