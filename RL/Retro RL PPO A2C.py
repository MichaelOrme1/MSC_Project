import retro
from baselines.common.retro_wrappers import *
from stable_baselines3 import PPO
from typing import Callable
from stable_baselines3.common.logger import *

env = retro.make(game = "SuperMarioWorld-Snes",use_restricted_actions=retro.Actions.DISCRETE,state = 'YoshiIsland1',obs_type = retro.Observations.IMAGE)


# set up logger
new_logger = configure(folder="./PPO/Image", format_strings=["stdout", "csv"])





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



model = PPO("CnnPolicy",env,verbose=1,learning_rate=linear_schedule(0.001))
model.set_logger(new_logger)
model.learn(total_timesteps=1000000,log_interval=4)
model.save("PPO_SMW_IMAGE")

env.close()
env = retro.make(game = "SuperMarioWorld-Snes",use_restricted_actions=retro.Actions.DISCRETE,state = 'YoshiIsland1',obs_type = retro.Observations.IMAGE)

new_logger = configure(folder="./A2C/Image", format_strings=["stdout", "csv"])
#Frame skip (hold an action for this many frames) and sticky actions
env = StochasticFrameSkip(env,4,0.25)
#scale and turn RGB image to grayscale
env = WarpFrame(env,width=84,height=84,grayscale=True)


model = A2C("CnnPolicy",env,verbose=1,learning_rate=linear_schedule(0.001))
model.set_logger(new_logger)
model.learn(total_timesteps=1000000,log_interval=4)
model.save("A2C_SMW_IMAGE")

##obs = env.reset()
##while True:
##    action, _states = model.predict(obs, deterministic=True)
##    obs, reward, done, info = env.step(action)
##    env.render()
##    if done:
##      obs = env.reset()
