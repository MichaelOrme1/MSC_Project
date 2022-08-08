import retro
import numpy as np
import tensorflow as tf
from tensorflow import keras
from baselines.common.retro_wrappers import *

model = tf.keras.models.load_model('./Keras_Models/sigmoid_test_IMG_1session5')
env = retro.make(game = "SuperMarioWorld-Snes",state='YoshiIsland2',obs_type = retro.Observations.IMAGE)

print(model.input.shape)
#Frame skip (hold an action for this many frames) and sticky actions
env = StochasticFrameSkip(env,4,0.25)
#scale and turn RGB image to grayscale
env = WarpFrame(env,width=84,height=84,grayscale=True)
obs = env.reset()

blank_action = np.zeros(12)

while True:
    obs = np.expand_dims(obs,0)#Add virtual batch
    output = model.predict(obs)#Predict based on observations
   
    for action in output:
        print(action)
        action = np.around(action)
        
        action = action.astype(int)
        #i=-1
        #print(action)
        #for actions in action:
            #i=i+1
            #if actions >= 0.5:
                #action[i] = 1
            #else:
                #action[i] = 0
             
                
    #action = action.astype(int)
    print(action)
    obs, rew, done, _info = env.step(action)#Use generated action
    obs, rew, done, _info = env.step(blank_action)#Stop it from holding an action, like jump
    env.render()
    

   
   
    
    
    if done:
        obs=env.reset()
   

