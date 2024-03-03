import numpy as np
from collections import deque, Counter
import sys
from reward_radar_algo_final import RewCal_General
# import overall_func
sys.path.append('../../signal_simulate')
from load_func import load_func



class Jammer_single:
    def __init__(self, num_sf, num_sp, env_seed):
        np.random.seed(env_seed)
        self.num_sf = num_sf
        self.num_sp = num_sp
        self.nb_actions_radar = np.power(self.num_sf, self.num_sp)
        self.nb_actions_jammer = self.num_sf
        self.actions_radar = list(range(self.nb_actions_radar))
        self.actions_jammer = list(range(self.nb_actions_jammer))
        self.totalfreq_history = []

    def step_BLN(self, a_radar, type, interact_version=0):
        if type == 'det0':
            a_jammer, strategy_jammer = self.jammer_det0()  # K=0, Deterministic
        elif type == 'sta0':
            a_jammer, strategy_jammer = self.jammer_sta0()  # K=0, Stationary
        else:
            jam_type = type[:-1]
            select_len = int(type[-1])
            if jam_type == 'det':
                a_jammer, strategy_jammer = self.jammer_det(select_len)  # deterministic
            elif jam_type == 'freq':
                a_jammer, strategy_jammer = self.jammer_freq(select_len)  # freq, stationary
            elif jam_type == 'nofreq':
                a_jammer, strategy_jammer = self.jammer_nofreq(select_len)  # nofreq, stationary
        self.radar_action = self.Num2Act_Radar(a_radar)
        self.totalfreq_history.extend(self.radar_action)

        if interact_version==0:
            reward = RewCal_General(self.num_sf, self.num_sp).get_reward(a_radar, a_jammer)
        elif interact_version==1:
            radar_action_v = self.Num2Act_Radar(a_radar)
            radar_action_v = np.array(radar_action_v)
            jammer_action_v = np.array([1.0, a_jammer + 1.]).reshape(1, -1)
            reward,_,_,_=load_func(self.path,self.eng,radar_action_v, jammer_action_v)
        else:
            assert interact_version==1 | interact_version==0

        reward_vec = RewCal_General(self.num_sf, self.num_sp).get_reward_vector(a_jammer)
        reward_matrix = RewCal_General(self.num_sf, self.num_sp).get_reward_matrix()
        reward_exp = np.matmul(reward_matrix, strategy_jammer)

        return reward, reward_vec, reward_exp, a_jammer, strategy_jammer

    def step_RL(self, a_radar, type):
        if type == 'det0':
            a_jammer, strategy_jammer = self.jammer_det0()  # K=0, Deterministic
        elif type == 'sta0':
            a_jammer, strategy_jammer = self.jammer_sta0()  # K=0, Stationary
        else:
            jam_type = type[:-1]
            select_len = int(type[-1])
            if jam_type == 'det':
                a_jammer, strategy_jammer = self.jammer_det(select_len)  # deterministic
            elif jam_type == 'freq':
                a_jammer, strategy_jammer = self.jammer_freq(select_len)  # freq, stationary
            elif jam_type == 'nofreq':
                a_jammer, strategy_jammer = self.jammer_nofreq(select_len)  # nofreq, stationary

        self.radar_action = self.Num2Act_Radar(a_radar)
        self.totalfreq_history.extend(self.radar_action)
        return a_jammer




    def Num2Act_Radar(self, a):
        x = []
        for _ in range(self.num_sp):
            x.append(a % self.num_sf)  # Get the remainder of y when divided by 10
            a = a // self.num_sf  # Divide y by 10 to move to the next digit
        x.reverse()  # Reverse the order of elements in x
        return x

    def Act2Num_Jammer(self, act_j):

        return act_j[0]

    def Freqs_Static(self, history):  # Static on each frequency on a_radar specific history
        freq_static = {i: 0 for i in range(self.num_sf)}
        collect = Counter(history)
        for f in collect.keys():
            freq_static[f] = collect[f]

        return freq_static

    def jammer_det0(self): # k=0, deterministic
        act_jammer = [1]*self.num_sp # Focus on the second frequency
        a_jammer = self.Act2Num_Jammer(act_jammer)
        strategy_jammer = np.zeros(self.nb_actions_jammer)
        strategy_jammer[a_jammer] = 1

        return a_jammer, strategy_jammer

    def jammer_sta0(self): # stationary opponent
        freq_prob = np.zeros(self.num_sf)
        freq_prob[0] = 0.6
        freq_prob[1] = 0.4
        a_jammer = np.random.choice(np.arange(self.num_sf), 1, p=freq_prob)[0] # focus on first two frequencies
        strategy_jammer = freq_prob

        return a_jammer, strategy_jammer

    def jammer_det(self, select_len): # History-dependent deterministic jammer
        select_his = select_len*self.num_sp
        if len(self.totalfreq_history) < select_his:
            act_jammer = [0]*self.num_sp
        else:
            history = self.totalfreq_history[-1 * select_his:]
            freq_static = self.Freqs_Static(history)  # Static on each frequency on this specific history
            # Find the most common frequency
            largest_freq = max(freq_static, key=lambda k: freq_static[k])
            act_jammer = [largest_freq]*self.num_sp
        a_jammer = self.Act2Num_Jammer(act_jammer)
        strategy_jammer = np.zeros(self.nb_actions_jammer)
        strategy_jammer[a_jammer] = 1

        return a_jammer, strategy_jammer

    def jammer_freq(self, select_len): # History-dependent freq stationary jammer
        select_his = select_len * self.num_sp
        if len(self.totalfreq_history) < select_his:
            act_jammer = [0] * self.num_sp
            strategy_jammer = np.zeros(self.nb_actions_jammer)
            strategy_jammer[0] = 1
        else:
            history = self.totalfreq_history[-1 * select_his:]
            freq_static = self.Freqs_Static(history)  # Static on each frequency on this specific history
            sorted_freqs = sorted(freq_static, key=freq_static.get, reverse=True)
            largest_two_freqs = sorted_freqs[:2]
            freq_prob = np.zeros(self.num_sf)
            freq_prob[largest_two_freqs[0]] = 0.7
            freq_prob[largest_two_freqs[1]] = 0.3
            act = np.random.choice(np.arange(self.num_sf), 1, p=freq_prob)[0]  # focus on two most common frequencies
            act_jammer = [act] * self.num_sp
            strategy_jammer = freq_prob
        a_jammer = self.Act2Num_Jammer(act_jammer)


        return a_jammer, strategy_jammer


    def jammer_nofreq(self, select_len): # History-dependent nofreq stationary jammer
        select_his = select_len * self.num_sp
        if len(self.totalfreq_history) < select_his:
            act_jammer = [0] * self.num_sp
            strategy_jammer = np.zeros(self.nb_actions_jammer)
            strategy_jammer[0] = 1
        else:
            history = self.totalfreq_history[-1 * select_his:]
            freq_static = self.Freqs_Static(history)  # Static on each frequency on this specific history
            static = list(freq_static.values())
            freq_prob = np.array(static) / np.array(static).sum()
            act = np.random.choice(np.arange(self.num_sf), 1, p=freq_prob)[0]
            act_jammer = [act] * self.num_sp
            strategy_jammer = freq_prob
        a_jammer = self.Act2Num_Jammer(act_jammer)

        return a_jammer, strategy_jammer







