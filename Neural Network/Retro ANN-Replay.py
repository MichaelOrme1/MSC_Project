import retro
import numpy as np
import tensorflow as tf
from tensorflow import keras
from baselines.common.retro_wrappers import *

model = tf.keras.models.load_model('./Keras_Models/softmax_test_IMG_1session')
env = retro.make(game = "SuperMarioWorld-Snes",state='YoshiIsland2',obs_type = retro.Observations.IMAGE)

#Frame skip (hold an action for this many frames) and sticky actions
env = StochasticFrameSkip(env,4,0.25)
#scale and turn RGB image to grayscale
env = WarpFrame(env,width=84,height=84,grayscale=True)
obs = env.reset()



while True:
    obs = np.expand_dims(obs,0)#Add virtual batch
    output = model.predict(obs)#Predict based on observations
   
    for action in output:
        i=-1
        #action=action.numpy()
        #print(["B", "Y", "SELECT", "START", "UP", "DOWN", "LEFT", "RIGHT", "A", "X", "L", "R"])
        print(action)
        chosen = np.random.choice(action,1,p=action)
        for c in chosen:
            #print(c)
            position = np.where(action==c)
            action[position] = 1
        #print(chosen)
        #action[chosen] = 1
        for actions in action:
            
            i=i+1
            #actions = int(actions)
            if actions != 1:   
                action[i] = 0
             
                
    action = action.astype(int)
    print(action)
    obs, rew, done, _info = env.step(action)#Use generated action 
    env.render()
    

   
   
    
    
    if done:
        break
   

