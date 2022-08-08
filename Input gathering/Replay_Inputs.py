
import numpy as np
import retro
import time

env = retro.make(game = "SuperMarioWorld-Snes",state='YoshiIsland2',obs_type = retro.Observations.IMAGE)

Training = np.load("TrainingData_IMG_1session.npy",allow_pickle=True)
Xtrain = Training[0]
ytrain = Training[1]

ytrain =  np.array([np.array(val) for val in ytrain])#Fixes issues with numpy loading

ytrain =  np.array([val.reshape(1,12) for val in ytrain])#Reshape to fit model


print(ytrain.shape)


obs = env.reset()

for action in ytrain:
    for a in action:
   
        obs, rew, done, _info = env.step(a)#Use generated action
        #time.sleep(1)
        env.render()

