import retro
from baselines.common.retro_wrappers import *
from stable_baselines3 import DQN
from typing import Callable
#from stable_baselines3.common.monitor import Monitor
#from stable_baselines3.common.callbacks import CheckpointCallback, EveryNTimesteps
#from stable_baselines3.common import logger
from stable_baselines3.common.logger import *

env = retro.make(game = "SuperMarioWorld-Snes",use_restricted_actions=retro.Actions.DISCRETE,state = 'YoshiIsland1',obs_type = retro.Observations.IMAGE)


# set up logger
new_logger = configure(folder="./DQN/Image_Test", format_strings=["stdout", "csv"])



#checkpoint_on_event = CheckpointCallback(save_freq=1, save_path='./logs/')
#event_callback = EveryNTimesteps(n_steps=500, callback=checkpoint_on_event)

#Frame skip (hold an action for this many frames) and sticky actions
env = StochasticFrameSkip(env,4,0.25)
#scale and turn RGB image to grayscale
env = WarpFrame(env,width=84,height=84,grayscale=True)

def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        return progress_remaining * initial_value

    return func


#make_output_format(_format = "csv",log_dir = ".")
model = DQN("CnnPolicy",env,verbose=1,buffer_size=10000,learning_rate=linear_schedule(0.001))
model.set_logger(new_logger)
model.learn(total_timesteps=250000,log_interval=4)
model.save("DQN_SMW_IMAGE_Test")

#nv.close()

#env = retro.make(game = "SuperMarioWorld-Snes",use_restricted_actions=retro.Actions.DISCRETE,state = 'YoshiIsland1',obs_type = retro.Observations.RAM)

#new_logger = configure(folder="./DQN/RAM", format_strings=["stdout", "csv"])


#Frame skip (hold an action for this many frames) and sticky actions
#env = StochasticFrameSkip(env,4,0.25)

#make_output_format(_format = "csv",log_dir = ".")
#model = DQN("MlpPolicy",env,verbose=1,buffer_size=10000,learning_rate=linear_schedule(0.001))
#model.set_logger(new_logger)
#model.learn(total_timesteps=1000000,log_interval=4)
#model.save("DQN_SMW_RAM")

#env.close()

#env = retro.make(game = "SuperMarioWorld-Snes",use_restricted_actions=retro.Actions.DISCRETE,state = 'YoshiIsland1',obs_type = retro.Observations.RAM2)
#new_logger = configure(folder="./DQN/RAM2", format_strings=["stdout", "csv"])


#Frame skip (hold an action for this many frames) and sticky actions
#env = StochasticFrameSkip(env,4,0.25)

#make_output_format(_format = "csv",log_dir = ".")
#model = DQN("MlpPolicy",env,verbose=1,buffer_size=10000,learning_rate=linear_schedule(0.001))
#model.set_logger(new_logger)
#model.learn(total_timesteps=1000000,log_interval=4)
#model.save("DQN_SMW_RAM2")

##obs = env.reset()
##while True:
##    action, _states = model.predict(obs, deterministic=True)
##    obs, reward, done, info = env.step(action)
##    env.render()
##    if done:
##      obs = env.reset()
