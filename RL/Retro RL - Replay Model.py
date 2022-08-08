import retro
from baselines.common.retro_wrappers import *
from stable_baselines3 import DQN,PPO,A2C
from stable_baselines3.common.policies import obs_as_tensor
import time
import numpy as np





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
values = []
reward_count = 0
reward_total = 0
reward_array = []
reward_count_array = []

env = retro.make(game = "SuperMarioWorld-Snes",use_restricted_actions=retro.Actions.DISCRETE,state='YoshiIsland1',obs_type = retro.Observations.RAM)
#Frame skip (hold an action for this many frames) and sticky actions
env = StochasticFrameSkip(env,4,0.25)
#scale and turn RGB image to grayscale
#env = WarpFrame(env,width=84,height=84,grayscale=True)
#blank_action = 0



model = DQN.load("DQN_SMW_RAM")



#start = time.time()
obs = env.reset()

print(obs.shape)




while True:
    action, _states = model.predict(obs, deterministic=False)
    
    obs, reward, done, info = env.step(action)
    #obs, rew, done, info = env.step(blank_action)#Stop it from holding an action, like jump
    
 
    print(env.get_action_meaning(action))
##    for actions in env.get_action_meaning(action):
##        if actions == "B":
##            B_Value = B_Value+1
##        if actions == "Y":
##            Y_Value = Y_Value+1
##        if actions == "SELECT":
##            SELECT_Value = SELECT_Value+1
##        if actions == "START":
##            START_Value = START_Value+1
##        if actions == "UP":
##            UP_Value = UP_Value+1
##        if actions == "DOWN":
##            DOWN_Value = DOWN_Value+1
##        if actions == "LEFT":
##            LEFT_Value = LEFT_Value+1
##        if actions == "RIGHT":
##            RIGHT_Value = RIGHT_Value+1
##        if actions == "A":
##            A_Value = A_Value+1
##        if actions == "X":
##            X_Value = X_Value+1
##        if actions == "L":
##            L_Value = L_Value+1
##        if actions == "R":
##            R_Value = R_Value+1
##  
    env.render()
##    values =[B_Value,Y_Value,SELECT_Value,START_Value,UP_Value,DOWN_Value,LEFT_Value,RIGHT_Value,A_Value,X_Value,L_Value,R_Value]
    #print(values)
    if done:
      obs=env.reset()
