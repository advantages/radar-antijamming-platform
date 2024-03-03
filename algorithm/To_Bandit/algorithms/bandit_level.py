from collections import deque
import numpy as np
import torch
from .level_history_dependent import Hisdep
from .level_transfermer02 import HisNN
# from .reward_radar_algo import RewCal
import sys
sys.path.append('../../../jammer_environment')
from reward_radar_algo_final import RewCal_General

class Bandit_Level:
    """Differ opponents """

    def __init__(self, actions01, actions02, alg_seed, num_sf, num_sp, max_history, cuda_num):
        np.random.seed(alg_seed)
        torch.manual_seed(alg_seed)
        self.actions_radar = actions01
        self.actions_jammer = actions02
        self.nb_actions_radar = len(actions01)
        self.nb_actions_jammer = len(actions02)
        self.num_sp = num_sp
        self.num_sf = num_sf
        self.max_history = max_history  # max_history value is not covered (because we have k=0)
        self.num_model = 3
        self.select_history = np.ones(self.max_history) / self.max_history
        self.s_history = np.zeros(self.max_history)
        self.select_model = np.array([[1/3, 1/3, 1/3]]*self.max_history) # Matrix for different model and different history length
        self.s_model = np.array([[0., 0., 0.]] * self.max_history)
        self.str_radar = np.ones(self.nb_actions_radar) / self.nb_actions_radar
        self.str_radar_mix = np.ones((self.nb_actions_radar, self.max_history)) / self.nb_actions_radar
        self.str_radar_all = [np.ones((self.nb_actions_radar, self.num_model)) / self.nb_actions_radar] * self.max_history
        self.str_radar_all[0] = np.ones(self.nb_actions_radar) / self.nb_actions_radar
        self.T = 0
        self.iterCount=0
        # self.eta_his = 0.1
        # self.eta_model = 0.1
        self.recent_history = deque(maxlen=self.max_history)  # Largest stored history
        # self.recent_history = deque(maxlen=7)
        self.hisdep = Hisdep(self.actions_radar, self.actions_jammer, self.max_history, self.num_sf, self.num_sp)
        self.hisnn = HisNN(self.actions_radar, self.actions_jammer, self.max_history, self.num_sf, self.num_sp, cuda_num)
        self.A_hat = RewCal_General(self.num_sf, self.num_sp).get_reward_matrix()
        self.cuda_num = cuda_num

    def play(self):
        self.select_len = np.random.choice(np.arange(self.max_history), 1, p=self.select_history)[0] # Choose history length
        # self.select_len = 1
        if self.select_len == 0:
            self.str_radar = self.str_radar_all[0]
        else:
            # self.select_model = np.array([[0, 0, 1]]*self.max_history)
            self.model = np.random.choice(np.arange(self.num_model), 1, p=self.select_model[self.select_len, :])[0]
            # self.model=0
            self.str_radar = self.str_radar_all[self.select_len][:, self.model] # str_radar_mix constains strategies for all history length at this round
            flag_str = np.random.choice([0, 1], p=[0.95, 0.05])
            if flag_str == 1 and self. select_len > 2:
                self.str_radar = np.ones(self.nb_actions_radar) / self.nb_actions_radar
        self.a_radar = np.random.choice(self.actions_radar, 1, p=self.str_radar)[0]
        #print(self.a_radar)
        self.iterCount=self.iterCount+1
        #if self.iterCount<=6000:
        #    self.str_radar_all[0] = np.ones(self.nb_actions_radar) / self.nb_actions_radar
        #    self.a_radar=np.random.choice(self.actions_radar, 1, p=self.str_radar_all[0])[0]
        #else:
        #    self.a_radar = np.random.choice(self.actions_radar, 1, p=self.str_radar)[0]

        #self.str_radar_all[0] = np.ones(self.nb_actions_radar) / self.nb_actions_radar
        #print(self.str_radar_all[0])
        #self.a_radar=np.random.choice(self.actions_radar, 1, p=self.str_radar_all[0])[0]
        self.recent_history.append(self.a_radar)
        return self.a_radar

    def update_policy(self, a_jammer):
        self.T += 1
        self.update_hispick(a_jammer)
        self.update_modelpick(a_jammer)
        self.str_update(a_jammer)

    # Choosing history length
    def update_hispick(self, a_jammer): # Update first with a_jammer, then get new str_radar predicted for next round
        # Normalize A_hat
        # normalized_A_hat = self.A_hat / self.A_hat.max()
        # loss_A_hat = np.ones(self.A_hat.shape) - normalized_A_hat
        # loss_vector = loss_A_hat[:, a_jammer]
        # rew_vector = self.A_hat[:, a_jammer]
        rew_history = np.matmul(np.transpose(self.str_radar_mix), self.A_hat[:, a_jammer])
        r_history = rew_history.reshape(-1) - np.dot(np.array(self.select_history).reshape(-1), rew_history.reshape(-1))*np.ones(self.max_history)
        self.s_history += r_history
        sp_history = np.maximum(self.s_history, np.zeros(self.max_history))
        # self.s_history = np.maximum(self.s_history, np.zeros(self.max_history))
        # sp_history = self.s_history
        normalize = sp_history.sum()
        if normalize > 0:
            self.select_history = sp_history / normalize
        else:
            self.select_history = np.ones(self.max_history) / self.max_history

    def update_modelpick(self, a_jammer):
        for i in np.arange(1, self.max_history):
            rew_model = np.matmul(np.transpose(self.str_radar_all[i]), self.A_hat[:, a_jammer])
            r_model = rew_model.reshape(-1) - np.dot(self.select_model[i, :].reshape(-1), rew_model.reshape(-1)) * np.ones(
                len(rew_model))
            self.s_model[i, :] += r_model
            sp_model = np.maximum(self.s_model[i, :], np.zeros(len(r_model)))
            # self.s_model[i, :] = np.maximum(self.s_model[i, :], np.zeros(len(r_model)))
            # sp_model = self.s_model[i, :]
            normalize = sp_model.sum()
            if normalize > 0:
                self.select_model[i, :] = sp_model / normalize
            else:
                self.select_model[i, :] = np.ones(len(r_model)) / len(r_model)


    def str_update(self, a_jammer):
        # Two steps: 1) Predict new str_radar 2) Update all the record matrix
        # k=0, stationary model and look-though model
        str_radar_nohistory = self.hisdep.stationary(a_jammer, self.A_hat)
        self.str_radar_mix[:, 0] = str_radar_nohistory
        self.str_radar_all[0] = str_radar_nohistory
        # k>0, str_radar is ar weighted sum
        for i in np.arange(1, self.max_history):
        # for i in np.arange(1, 2):
        # for i in np.arange(3, 3):
            # freq-dependent model
            str_radar_freq = self.hisdep.history_dependent_freq(a_jammer, i, self.recent_history, self.A_hat)
            # str_radar_freq = np.ones(self.nb_actions_radar) / self.nb_actions_radar
            # no-prior model
            if i < 5:
                str_radar_nofreq = self.hisdep.history_dependent_nofreq(a_jammer, i, self.recent_history, self.A_hat)
            else:
                str_radar_nofreq = np.ones(self.nb_actions_radar) / self.nb_actions_radar
            # neural network
            if i > 2:
                str_radar_nn = self.hisnn.neural_network_predict(a_jammer, i, self.recent_history, self.A_hat)
            else:
                str_radar_nn = np.ones(self.nb_actions_radar) / self.nb_actions_radar
            # Mixed strategy
            str_radar_total = np.array([str_radar_freq, str_radar_nofreq, str_radar_nn])
            str_radar_mix = np.dot(np.transpose(str_radar_total), self.select_model[i, :].reshape(-1))
            self.str_radar_mix[:, i] = str_radar_mix
            self.str_radar_all[i] = np.transpose(str_radar_total)

            # Update select_model probability conditioned on previous str_radar predicted by each model
            # rew_model = np.matmul(str_radar_total, self.A_hat[:, a_jammer])
            # r_model = rew_model.reshape(-1) - np.dot(self.select_model[i,:].reshape(-1), rew_model.reshape(-1)) * np.ones(
            #     len(rew_model))
            # self.s_model[i,:] += r_model
            # sp_model = np.maximum(self.s_model[i,:], np.zeros(len(r_model)))
            # normalize = sp_model.sum()
            # if normalize > 0:
            #     self.select_model[i,:] = sp_model / normalize
            # else:
            #     self.select_model[i,:] = np.ones(len(r_model)) / len(r_model)

            # for j in range(len(self.s_model[i, :])):
            #     self.s_model[i, j] += self.eta_model * np.dot(str_radar_total[j, :], rew_vector)
            # self.select_model[i, :] = softmax(self.s_model[i, :])

        # # Normalization
        # row_sums = self.s_model.sum(axis=1)
        # self.select_model = self.s_model/row_sums[:, np.newaxis]

    # def softmax(self, x):
    #     """Compute softmax values for each sets of scores in x."""
    #     e_x = np.exp(x)
    #     return e_x / e_x.sum()

