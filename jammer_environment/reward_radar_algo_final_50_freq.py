import numpy as np
from scipy.io import loadmat



class RewCal_General:
    def __init__(self, num_sf, num_sp):
        self.num_sp = num_sp
        self.num_sf = num_sf
        self.nb_actions_radar = np.power(num_sf, num_sp)
        self.nb_actions_jammer = num_sf
        # mat = loadmat('/home/lqliu/MatrixGame_ptp_cp/algorithms/reward_4_4.mat')
        # mat = loadmat('/home/lqliu/MatrixGame_ptp_cp/algorithms/reward.mat')
        # self.reward_matrix = mat['A']
        self.power = [4.835,6.269,7.545,8.401,9.218,10.109,11.086,12.076,13.057,14.029,14.991,15.932,16.812,17.724,18.505,19.321,19.919,
                      20.660,21.179,21.794,22.193,22.559,22.831,22.697,22.129,21.465,20.705,19.587,18.191,16.810,15.490,14.110,12.872,11.588,
                      10.475,9.322,8.273,7.244,6.315,5.439,4.636,3.906,3.250,2.662,2.150,1.724,1.376,1.087,0.818,0.579]
        #print(self.reward_matrix)
        self.reward_matrix=self.init_matrix()

    def get_reward(self, ar, aj):
        reward = self.reward_matrix[ar, aj]
        return reward



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
