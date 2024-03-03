import numpy as np
from scipy.io import loadmat
import os
import overall_func

import sys
sys.path.append('../../signal_simulate')
from load_func import load_func



class RewCal_General:
    def __init__(self, num_sf, num_sp):
        self.num_sp = num_sp
        self.num_sf = num_sf
        self.nb_actions_radar = np.power(num_sf, num_sp)
        self.nb_actions_jammer = num_sf

        if os.path.exists('./reward_{}_{}.mat'.format(int(num_sf), int(num_sp))):
            mat = loadmat('./reward_{}_{}.mat'.format(int(num_sf), int(num_sp)))
            self.reward_matrix = mat['A']
        else:
            self.eng = overall_func.initialize()
            self.path = '../../global_param/setting.json'
            self.power = self.generate_power()
            self.reward_matrix=self.init_matrix()

    def get_reward(self, ar, aj):
        reward = self.reward_matrix[ar, aj]
        return reward

    def generate_power(self):
        power=[]
        for i in range(self.num_sf):
            act=0
            for j in range(self.num_sp):
                act+=i*pow(self.num_sf,self.num_sp-j-1)
            radar_action_v = self.Num2Act_Radar(act)
            radar_action_v = np.array(radar_action_v)
            if i <= (0.5 * self.num_sf):
                jammer_action_v = np.array([1.0, i + int(self.num_sf / 4) + 1.]).reshape(1, -1)
            else:
                jammer_action_v = np.array([1.0, i - int(self.num_sf / 4) + 1.]).reshape(1, -1)
            reward, _, _, _ = load_func(self.path, self.eng, radar_action_v, jammer_action_v)
            reward/=self.num_sp
            power.append(reward)
        return power

    def Num2Act_Radar(self, a):
        x = []
        for _ in range(self.num_sp):
            x.append(a % self.num_sf)  # Get the remainder of y when divided by 10
            a = a // self.num_sf  # Divide y by 10 to move to the next digit
        x.reverse()  # Reverse the order of elements in x
        return x


    def get_reward_for_gene(self, ar, aj):
        act_r = self.Num2Act_Radar(ar)
        act_j = self.Num2Act_Jammer(aj)
        # reward = self.reward_matrix[ar, aj]
        reward=0
        for i in range(self.num_sp):
            if act_r[i]==act_j[i]:
                continue
            else:
                reward+=self.power[act_r[i]]
        return reward

    def get_reward_vector(self, aj):
        # For each radar's a_radar, get the exact reward
        reward_vector = self.reward_matrix[:, aj]

        return reward_vector # np.array

    def init_matrix(self):
        reward_matrix = np.zeros((pow(self.num_sf, self.num_sp), self.num_sf))

        for i in range(pow(self.num_sf, self.num_sp)):
            for j in range(self.num_sf):
                reward_matrix[i, j] = self.get_reward_for_gene(i, j)
        return reward_matrix


    def get_reward_matrix(self):

        reward_matrix = self.reward_matrix

        return reward_matrix # np.array


    def Num2Act_Radar(self, ar):
        x = []
        for _ in range(self.num_sp):
            x.append(ar % self.num_sf)  # Get the remainder of y when divided by 10
            ar = ar // self.num_sf  # Divide y by 10 to move to the next digit
        x.reverse()  # Reverse the order of elements in x
        return x

    def Num2Act_Jammer(self, aj):
        x = [aj]*self.num_sp

        return x




    #
    # def get_jamming(self, aj):
    #     # power allocated for [f0, f1, f2]
    #     # Divided as [1, (1/3)*inr, (1/2)*inr, inr]
    #     jamming_freq = [1, 1, 1]
    #     if aj[-1] == 1:
    #         jamming_freq[aj[0]] = self.inr
    #     elif aj[-1] == 2:
    #         jamming_freq[aj[0]] = (1 / 2) * self.inr
    #         jamming_freq[aj[1]] = (1 / 2) * self.inr
    #     elif aj[-1] == 3:
    #         jamming_freq = [(1 / 3) * self.inr, (1 / 3) * self.inr, (1 / 3) * self.inr]
    #     else:
    #         pass
    #     return jamming_freq
    #
    # def get_radar_reward(self, a_radar, jamming_freq):
    #     subpluse0 = self.snr[a_radar[0]] / jamming_freq[a_radar[0]]
    #     subpluse1 = self.snr[a_radar[1]] / jamming_freq[a_radar[1]]
    #     subpluse2 = self.snr[a_radar[2]] / jamming_freq[a_radar[2]]
    #     radar_reward = subpluse0 + subpluse1 + subpluse2
    #
    #     return radar_reward
    #
    # def get_seperate(self, a_radar, jamming_freq):
    #     subpulse_snr = []
    #     subpulse_inr = []
    #     for i in range(3):
    #         if jamming_freq[a_radar[i]] == 1:
    #             subpulse_snr.append(self.snr[a_radar[i]])
    #             subpulse_inr.append(0)
    #         else:
    #             subpulse_snr.append(0)
    #             subpulse_inr.append(self.snr[a_radar[i]] / jamming_freq[a_radar[i]])
    #
    #     reward_snr = sum(subpulse_snr)
    #     reward_inr = sum(subpulse_inr)
    #
    #     return reward_snr, reward_inr
    #
    # def Num2Act_Radar(self, a_radar):
    #     z, z1 = a_radar % 3, a_radar // 3
    #     if z1 < 3:
    #         y, x = z1, 0
    #     else:
    #         y = z1 % 3
    #         x = a_radar // 9
    #     return [x, y, z]
    #
    # def Num2Act_Jammer(self, aj):
    #     action_set = [[0, 1], [1, 1], [2, 1], [0, 1, 2], [0, 2, 2], [1, 2, 2], [3]]
    #
    #     return action_set[aj]
