import gc
import gym
import gzip
import gym.spaces
import json
import numpy as np
import os
import retro
import retro.data
from gym.utils import seeding
from itertools import product
#from rominfo import *
"""
The MIT License

Copyright (c) 2017-2018 OpenAI (http://openai.com)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""
gym_version = tuple(int(x) for x in gym.__version__.split('.'))

__all__ = ['RetroEnv']


class RetroEnv(gym.Env):
    """
    Gym Retro environment class

    Provides a Gym interface to classic video games
    """
    metadata = {'render.modes': ['human', 'rgb_array'],
                'video.frames_per_second': 60.0}

    def __init__(self, game, state=retro.State.DEFAULT, scenario=None, info=None, use_restricted_actions=retro.Actions.FILTERED,
                 record=True, players=1, inttype=retro.data.Integrations.STABLE, obs_type=retro.Observations.IMAGE):
        if not hasattr(self, 'spec'):
            self.spec = None
        self._obs_type = obs_type
        self.img = None
        self.ram = None
        self.state = np.zeros((13,13))
        self.viewer = None
        self.gamename = game
        self.statename = state
        self.initial_state = None
        self.players = players

        metadata = {}
        rom_path = retro.data.get_romfile_path(game, inttype)
        metadata_path = retro.data.get_file_path(game, 'metadata.json', inttype)

        if state == retro.State.NONE:
            self.statename = None
        elif state == retro.State.DEFAULT:
            self.statename = None
            try:
                with open(metadata_path) as f:
                    metadata = json.load(f)
                if 'default_player_state' in metadata and self.players <= len(metadata['default_player_state']):
                    self.statename = metadata['default_player_state'][self.players - 1]
                elif 'default_state' in metadata:
                    self.statename = metadata['default_state']
                else:
                    self.statename = None
            except (IOError, json.JSONDecodeError):
                pass

        if self.statename:
            self.load_state(self.statename, inttype)

        self.data = retro.data.GameData()

        if info is None:
            info = 'data'

        if info.endswith('.json'):
            # assume it's a path
            info_path = info
        else:
            info_path = retro.data.get_file_path(game, info + '.json', inttype)

        if scenario is None:
            scenario = 'scenario'

        if scenario.endswith('.json'):
            # assume it's a path
            scenario_path = scenario
        else:
            scenario_path = retro.data.get_file_path(game, scenario + '.json', inttype)

        self.system = retro.get_romfile_system(rom_path)

        # We can't have more than one emulator per process. Before creating an
        # emulator, ensure that unused ones are garbage-collected
        gc.collect()
        self.em = retro.RetroEmulator(rom_path)
        self.em.configure_data(self.data)
        self.em.step()

        core = retro.get_system_info(self.system)
        self.buttons = core['buttons']
        self.num_buttons = len(self.buttons)

        try:
            assert self.data.load(info_path, scenario_path), 'Failed to load info (%s) or scenario (%s)' % (info_path, scenario_path)
        except Exception:
            del self.em
            raise

        self.button_combos = self.data.valid_actions()
        if use_restricted_actions == retro.Actions.DISCRETE:
            combos = 1
            for combo in self.button_combos:
                combos *= len(combo)
            self.action_space = gym.spaces.Discrete(combos ** players)
        elif use_restricted_actions == retro.Actions.MULTI_DISCRETE:
            self.action_space = gym.spaces.MultiDiscrete([len(combos) if gym_version >= (0, 9, 6) else (0, len(combos) - 1) for combos in self.button_combos] * players)
        else:
            self.action_space = gym.spaces.MultiBinary(self.num_buttons * players)

        kwargs = {}
        if gym_version >= (0, 9, 6):
            kwargs['dtype'] = np.uint8
        
        if self._obs_type == retro.Observations.RAM:
            shape = self.get_ram().shape
        elif self._obs_type == retro.Observations.IMAGE:
            img = [self.get_screen(p) for p in range(players)]
            shape = img[0].shape
        elif self._obs_type == retro.Observations.RAM2:
            shape = (13,13)
            
            
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=shape, **kwargs)

        self.use_restricted_actions = use_restricted_actions
        self.movie = None
        self.movie_id = 0
        self.movie_path = None
        if record is True:
            self.auto_record()
        elif record is not False:
            self.auto_record(record)
        self.seed()
        if gym_version < (0, 9, 6):
            self._seed = self.seed
            self._step = self.step
            self._reset = self.reset
            self._render = self.render
            self._close = self.close

    def _update_obs(self):
        if self._obs_type == retro.Observations.RAM:
            self.ram = self.get_ram()
            return self.ram
        elif self._obs_type == retro.Observations.IMAGE:
            self.img = self.get_screen()
            return self.img
        #New Code
        elif self._obs_type == retro.Observations.RAM2:
            ram = self.getRam()
            state, x, y = self.getInputs(ram)
            state = np.reshape(state, (13, 13))
            return self.state
        else:
            raise ValueError('Unrecognized observation type: {}'.format(self._obs_type))

    def getRam(self):
    
        return np.array(list(self.data.memory.blocks[8257536]))

    def action_to_array(self, a):
        actions = []
        for p in range(self.players):
            action = 0
            if self.use_restricted_actions == retro.Actions.DISCRETE:
                for combo in self.button_combos:
                    current = a % len(combo)
                    a //= len(combo)
                    action |= combo[current]
            elif self.use_restricted_actions == retro.Actions.MULTI_DISCRETE:
                ap = a[self.num_buttons * p:self.num_buttons * (p + 1)]
                for i in range(len(ap)):
                    buttons = self.button_combos[i]
                    action |= buttons[ap[i]]
            else:
                ap = a[self.num_buttons * p:self.num_buttons * (p + 1)]
                for i in range(len(ap)):
                    action |= int(ap[i]) << i
                if self.use_restricted_actions == retro.Actions.FILTERED:
                    action = self.data.filter_action(action)
            ap = np.zeros([self.num_buttons], np.uint8)
            for i in range(self.num_buttons):
                ap[i] = (action >> i) & 1
            actions.append(ap)
        return actions

    def step(self, a):
        if self.img is None and self.ram is None and self.state is None:
            raise RuntimeError('Please call env.reset() before env.step()')
 
        for p, ap in enumerate(self.action_to_array(a)):
            if self.movie:
                for i in range(self.num_buttons):
                    self.movie.set_key(i, ap[i], p)
            self.em.set_button_mask(ap, p)

        if self.movie:
            self.movie.step()
        self.em.step()
        self.data.update_ram()
        ob = self._update_obs()
        rew, done, info = self.compute_step()
        return ob, rew, bool(done), dict(info)

    def reset(self):
        if self.initial_state:
            self.em.set_state(self.initial_state)
        for p in range(self.players):
            self.em.set_button_mask(np.zeros([self.num_buttons], np.uint8), p)
        self.em.step()
        if self.movie_path is not None:
            rel_statename = os.path.splitext(os.path.basename(self.statename))[0]
            self.record_movie(os.path.join(self.movie_path, '%s-%s-%06d.bk2' % (self.gamename, rel_statename, self.movie_id)))
            self.movie_id += 1
        if self.movie:
            self.movie.step()
        self.data.reset()
        self.data.update_ram()
        return self._update_obs()

    def seed(self, seed=None):
        self.np_random, seed1 = seeding.np_random(seed)
        # Derive a random seed. This gets passed as a uint, but gets
        # checked as an int elsewhere, so we need to keep it below
        # 2**31.
        seed2 = seeding.hash_seed(seed1 + 1) % 2**31
        return [seed1, seed2]

    def render(self, mode='human', close=False):
        if close:
            if self.viewer:
                self.viewer.close()
            return

        img = self.get_screen() if self.img is None else self.img
        if mode == "rgb_array":
            return img
        elif mode == "human":
            if self.viewer is None:
                from gym.envs.classic_control.rendering import SimpleImageViewer
                self.viewer = SimpleImageViewer()
            self.viewer.imshow(img)
            return self.viewer.isopen

    def close(self):
        if hasattr(self, 'em'):
            del self.em

    def get_action_meaning(self, act):
        actions = []
        for p, action in enumerate(self.action_to_array(act)):
            actions.append([self.buttons[i] for i in np.extract(action, np.arange(len(action)))])
        if self.players == 1:
            return actions[0]
        return actions

    def get_ram(self):
        blocks = []
        for offset in sorted(self.data.memory.blocks):
            arr = np.frombuffer(self.data.memory.blocks[offset], dtype=np.uint8)
            blocks.append(arr)
        return np.concatenate(blocks)

    def get_screen(self, player=0):
        img = self.em.get_screen()
        x, y, w, h = self.data.crop_info(player)
        if not w or x + w > img.shape[1]:
            w = img.shape[1]
        else:
            w += x
        if not h or y + h > img.shape[0]:
            h = img.shape[0]
        else:
            h += y
        if x == 0 and y == 0 and w == img.shape[1] and h == img.shape[0]:
            return img
        return img[y:h, x:w]

    def load_state(self, statename, inttype=retro.data.Integrations.DEFAULT):
        if not statename.endswith('.state'):
                statename += '.state'

        with gzip.open(retro.data.get_file_path(self.gamename, statename, inttype), 'rb') as fh:
            self.initial_state = fh.read()

        self.statename = statename

    def compute_step(self):
        if self.players > 1:
            reward = [self.data.current_reward(p) for p in range(self.players)]
        else:
            reward = self.data.current_reward()
        done = self.data.is_done()
        return reward, done, self.data.lookup_all()

    def record_movie(self, path):
        self.movie = retro.Movie(path, True, self.players)
        self.movie.configure(self.gamename, self.em)
        if self.initial_state:
            self.movie.set_state(self.initial_state)

    def stop_record(self):
        self.movie_path = None
        self.movie_id = 0
        if self.movie:
            self.movie.close()
            self.movie = None

    def auto_record(self, path=None):
        if not path:
            path = os.getcwd()
        self.movie_path = path




    #New Code
    def getInputs(self,ram, radius=6):
      '''
      getInputs(ram): returns an nd.array of enemies, obstacles within a radius around the agent
      '''
  
      marioX, marioY, layer1x, layer1y = self.getXY(ram)
      sprites = self.getSprites(ram)

      # vector size
      maxlen = (radius*2+1)*(radius*2+1)
      inputs = np.zeros(maxlen, dtype=int)
      
      # cada bloco de imagem representa 16x16 pixesl
      # portanto tendo um x,y de referência do Mario
      # devemos camihar de 16 em 16
      window = (-radius*16, radius*16 + 1, 16)
      j = 0

      def withinLimits(idx, ds1, ds2, r, maxlen):
        return (idx%(2*r + 1) + ds2 < 2*r + 1) and (idx + ds1*(2*r + 1) + ds2 < maxlen)
      
      for dy, dx in product(range(*window), repeat=2):
        # verifica se tem obstáculo na posição x+dx, y+dy
        # o +8 é para começar a medir a partir do meio do Mario
        tile = self.getTile(marioX+dx+8, marioY+dy, ram)
        
        # O Mario está sempre no meio, 
        # deve checar se o y está dentro do limite
        if tile==1 and marioY+dy < 0x1B0:
          inputs[j] = 1
        
        # Para cada sprite
        if sprites is not None:
            for i in range(len(sprites)):
              # Se estiver dentro do bloco de 16 x 16 (-8, +8)
              distx = np.abs(sprites[i]['x'] - marioX - dx)
              disty = np.abs(sprites[i]['y'] - marioY - dy)
              size = sprites[i]['size']
              if distx <= 8 and disty <= 8:
                # se estiver dentro dos limites, insira -1
                for s1, s2 in product(range(size), repeat=2):
                  if withinLimits(j, s1, s2, radius, maxlen):
                    inputs[j + s1*(radius*2 + 1) + s2] = -1
            j = j + 1
      return inputs, marioX, marioY



    def getSprites(self,ram):
      '''
      getSprites(ram): returns the sprites (blocks, enemies, items) displayed on the screen.
      '''
      
      sprites = []
      extsprites = []
      
      # There can be up to 12 sprites on the screen
      for slot in range(12):
        # if the status is 0, there is no sprite in that slot
        status = ram[0x14C8+slot]
        if status != 0:
          # x,y position of the sprite
          spriteX    = ram[0xE4+slot] + ram[0x14E0+slot]*256
          spriteY    = ram[0xD8+slot] + ram[0x14D4+slot]*256
          
          spriteSize = ram[0x0420+ram[0x15EA+slot]]  # sprite size
          spriteId   = ram[0x15EA+slot]              # what is the sprite?
          
          # if it is item (44) or block ? (216), do not insert in the information
          if spriteId != 44 and spriteId != 216:
            # either it is 1x1 or 4x4 blocks of our window
            size = 1
            if spriteSize == 0:
              size = 4
            sprites.append({'x': spriteX, 'y': spriteY, 'size': size})
          
          return sprites


    def getTile(self,dx,dy,ram):
      '''
      getTile(dx, dy, ram): retorna se tem um bloco que o mario possa pisar na posição dx, dy
      '''
      x = np.floor(dx/16)
      y = np.floor(dy/16)
  
      # 0x1C800 indica para cada pixel se é um obstáculo ou não
      # como obter o ponto certo foi retirado daqui: https://www.smwcentral.net/?p=viewthread&t=78887
      # return ram[0x1C800 + np.int(np.floor(x/16)*432 + y*16 + x%16)]

      # O endereço correto é 0x1F000, contribuição de Fernando Teixeira
      if (0x1F000 + np.int(np.floor(x/16)*432 + y*16 + x%16)) > 131071:
    
          return ram[131071]
      else:
          return ram[0x1F000 + np.int(np.floor(x/16)*432 + y*16 + x%16)]
    
        




    def getXY(self,ram):
        '''
        getXY(ram): returns agent position information
        though layer1? is not currently used, it may be useful with some
        learning algorithm changes.
        '''
        
        # x, y coordinates with respect to the entire phase 
        # They are stored in 2 bytes each
        # in little endian format
        marioX = ram[0x95]*256 + ram[0x94]
        marioY = ram[0x97]*256 + ram[0x96]
        
        # Coordinate of the visible part of the site
        layer1x = ram[0x1B]*256 + ram[0x1A]
        layer1y = ram[0x1D]*256 + ram[0x1C]
        
        return marioX.astype(np.int16), marioY.astype(np.int16), layer1x.astype(np.int16), layer1y.astype(np.int16)

