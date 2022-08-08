import retro
import numpy as np
from numpy import set_printoptions
import time

env = retro.make(game='SuperMarioWorld-Snes')
env.reset()
B_Value = 0
Y_Value = 0
SELECT_Value = 0
START_Value = 0
UP_Value = 0
DOWN_Value = 0
LEFT_Value = 0
RIGHT_Value = 0
A_Value = 0
X_Value = 0
L_Value = 0
R_Value = 0
reward_count = 0
reward_total = 0
reward_array = []
reward_count_array = []
start = time.time()
while True:
    action = env.action_space.sample()
    print((action))
    obs, rew, done, _info = env.step(action)
    env.render()
    #print(_obs.shape)
    
    
    
    
    
    #if _rew != 0:
        #end = time.time()
        #reward_count = end - start
        #reward_count_array.append(reward_count)
        #reward_total = reward_total + _rew
        #reward_array.append(reward_total)
                
    
    
##    if action[0] == 1:
##        B_Value = B_Value +1
##
##    if action[1] == 1:
##        Y_Value = Y_Value +1
##
##    if action[2] == 1:
##        SELECT_Value = SELECT_Value +1
##
##    if action[3] == 1:
##        START_Value = START_Value +1
##
##    if action[4] == 1:
##        UP_Value = UP_Value +1
##
##    if action[5] == 1:
##        DOWN_Value = DOWN_Value +1
##
##    if action[6] == 1:
##        LEFT_Value = LEFT_Value +1
##
##    if action[7] == 1:
##        RIGHT_Value = RIGHT_Value +1
##
##    if action[8] == 1:
##        A_Value = A_Value +1
##
##    if action[9] == 1:
##        X_Value = X_Value +1
##
##    if action[10] == 1:
##        L_Value = L_Value +1
##
##    if action[11] == 1:
##        R_Value = R_Value +1
##    
   
    
    
    if done:
        #values = [B_Value,Y_Value,SELECT_Value,START_Value,UP_Value,DOWN_Value,LEFT_Value,RIGHT_Value,A_Value,X_Value,L_Value,R_Value]
        #print(values)
        #print(reward_array)
        #print(reward_count_array)
        break
   

# ob = image of screen at time of action
# rew = amount of reward that he earned from whatever in scenario file
# done = whether done condition met
# info = dict of all values set in data.json`

#[100000000000] 1
#[010000000000] 2
#[001000000000] 3
#[000100000000] 4
#[000010000000] 5
#[000001000000] 6 
#[000000100000] 7
#[000000010000] 8 
#[000000001000] 9
#[000000000100] 10
#[000000000010] 11
#[000000000001] 12

#["B", "Y", "SELECT", "START", "UP", "DOWN", "LEFT", "RIGHT", "A", "X", "L", "R"]
