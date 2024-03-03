import numpy as np
from scipy.io import loadmat


class RewCal_General:
    def __init__(self, num_sf, num_sp):
        self.num_sp = num_sp
        self.num_sf = num_sf
        self.nb_actions_radar = np.power(num_sf, num_sp)
        self.nb_actions_jammer = num_sf


        try:
            mat = loadmat('../../jammer_environment/reward_{}_{}.mat'.format(int(num_sf),int(num_sp)))
        except ValueError:
            print("The corresponding .mat file is not included")


        self.reward_matrix = mat['A']


    def get_reward(self, ar, aj):
        act_r = self.Num2Act_Radar(ar)
        act_j = self.Num2Act_Jammer(aj)
        reward = self.reward_matrix[ar, aj]

        return reward

    def get_reward_vector(self, aj):
        # For each radar's a_radar, get the exact reward
        reward_vector = self.reward_matrix[:, aj]

        return reward_vector # np.array

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

