"""
Classic cart-pole system implemented by Rich Sutton et al.
Copied from http://incompleteideas.net/sutton/book/code/pole.c
permalink: https://perma.cc/C9ZM-652R
"""
import math
from typing import Optional, Tuple, Union
import sys

import gymnasium as gym
from gymnasium import logger, spaces
from gymnasium.envs.classic_control import utils
from gymnasium.error import DependencyNotInstalled
from gymnasium.experimental.vector import VectorEnv
from gymnasium.vector.utils import batch_space
import numpy as np
import pygame
import matlab
import gymnasium as gym
from gymnasium import spaces
# import matlab.engine
from collections import Counter
import torch
from jammer_env import Jammer_single, RewCal_General
import overall_func
import sys
sys.path.append('../../signal_simulate')
from load_func import load_func


class RadarGameEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, episode=32,cpi_length=1,num_sf=3,num_sp=3,select_version=0,jammer_type='det1',history_len=3):
        self.cpi = cpi_length  # The size of the square grid
        self.jammer_type=jammer_type
        # self.observation_space = spaces.Dict(
        #     {
        #         "radar": spaces.Box(0, 1, shape=(num_sp*cpi_length,num_sf), dtype=int),
        #         "jammer": spaces.Box(0, 1, shape=(num_sp*cpi_length,num_sf), dtype=int),
        #     }
        # )
        self.history_len = history_len


        self.observation_space=spaces.Box(0,num_sf-1,shape=(2*self.history_len,1),dtype=float)
        # self.observation_space = spaces.Box(0, self.num_sf.__pow__(self.num_sp), shape=(2, 4), dtype=int)

        self.episode=episode
        self.length=0
        self.count=0

        self.history_state=np.zeros((2,self.history_len))
        self.history_reward=np.zeros((1,self.history_len))


        self.num_sf=num_sf
        self.cpi=cpi_length
        self.num_sp=num_sp

        self.action_space = spaces.Discrete(int(self.num_sf.__pow__(self.num_sp)))
        # self.action_space = spaces.Box(low=0, high=self.num_sf.__pow__(self.num_sp)-1, shape=(1,), dtype=np.float32)

        self.select_version=select_version
        self.JammingG = Jammer_single(self.num_sf, self.num_sp, 1234)

        self.RewardG = RewCal_General(self.num_sf, self.num_sp)

        if self.select_version==1:
            self.eng = overall_func.initialize()

        self.path='../../global_param/setting.json'

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode


        self.window = None
        self.clock = None
    def _get_obs(self):
        state=np.concatenate([self.radar_act,self.jammer_act])
        return  state
    def _get_info(self):
        return {
            "reward": self.reward
        }



    def Freqs_Static(self, history):  # Static on each frequency on a_radar specific history
        freq_static = {i: 0 for i in range(self.num_sf)}
        collect = Counter(history)
        for f in collect.keys():
            freq_static[f] = collect[f]

        return freq_static

    def Num2Act_Radar(self, a):
        x = []
        for _ in range(self.num_sp):
            x.append(a % self.num_sf)  # Get the remainder of y when divided by 10
            a = a // self.num_sf  # Divide y by 10 to move to the next digit
        x.reverse()  # Reverse the order of elements in x
        return x



    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.length=0
        self.history_state=np.zeros((2,self.history_len))
        self.history_reward=np.zeros((1,self.history_len))
        state = self.history_state.reshape(-1, 1)

        return state, {}



    def step(self, radar_action):
        # radar_action=int(np.round(radar_action))    # trpo


        # print(radar_action)
        # radar_action_trans=np.array([radar_action])[:,np.newaxis]

        radar_action_v=self.Num2Act_Radar(radar_action)
        radar_action_v = np.array(radar_action_v)


        if self.select_version==0:
            jammer_action=self.JammingG.step_RL(radar_action,self.jammer_type)
            reward=self.RewardG.get_reward(radar_action,jammer_action)

            self.jammer_act=np.array([jammer_action])[:,np.newaxis]
            self.radar_act=np.array([radar_action])[:,np.newaxis]
            self.reward=np.array([reward])[:,np.newaxis]



        if self.select_version==1:
            jammer_action = self.JammingG.step_RL(radar_action, self.jammer_type)
            jammer_action_v = np.array([1.0, jammer_action + 1.]).reshape(1, -1)
            reward,_,_,_ = load_func(self.path,self.eng,radar_action_v, jammer_action_v)

            self.jammer_act = np.array([jammer_action])[:, np.newaxis]
            self.radar_act = np.array([radar_action])[:, np.newaxis]
            self.reward = np.array([reward])[:, np.newaxis]


        info=self._get_info()
        observation=self._get_obs()
        self.history_state=np.concatenate([observation,self.history_state[:,:-1]],axis=1)
        self.history_reward=np.concatenate([self.reward,self.history_reward[:,:-1]],axis=1)
        self.reward=np.sum(self.history_reward)

        observation=self.history_state.reshape(-1,1)

        if self.length <= self.episode-2:
            self.length += 1
            done = False
        else:
            done = True
        return observation, reward,False,done, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        return 0


    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
        self.eng.quit()
