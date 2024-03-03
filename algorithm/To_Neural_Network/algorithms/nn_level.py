from collections import deque
import numpy as np
import torch
from .level_transfermer02 import HisNN
# from .reward_radar_algo import RewCal
import sys
sys.path.append('../../../jammer_environment')
from reward_radar_algo_final import RewCal_General
# from reward_radar_algo_final_20_freq import RewCal_General
# from reward_radar_algo_final_50_freq import RewCal_General


class nn_level:
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
        # self.hisdep = Hisdep(self.actions_radar, self.actions_jammer, self.max_history, self.num_sf, self.num_sp)
        self.hisnn = HisNN(self.actions_radar, self.actions_jammer, self.max_history, self.num_sf, self.num_sp, cuda_num)
        self.A_hat = RewCal_General(self.num_sf, self.num_sp).get_reward_matrix()
        self.cuda_num = cuda_num

    def play(self):
        self.select_history=[0,0,0,0,0,1,0]
        self.select_len = np.random.choice(np.arange(self.max_history), 1, p=self.select_history)[0] # Choose history length
        # self.select_len = 1
        if self.select_len == 0:
            self.str_radar = self.str_radar_all[0]
        else:
            # self.select_model = np.array([[0, 0, 1]]*self.max_history)
            self.model = np.random.choice(np.arange(self.num_model), 1, p=self.select_model[self.select_len, :])[0]
            self.model=2
            self.str_radar = self.str_radar_all[self.select_len][:, self.model] # str_radar_mix constains strategies for all history length at this round
            flag_str = np.random.choice([0,1], p=[0.9, 0.1])
            if flag_str == 1:
                self.str_radar = np.ones(self.nb_actions_radar) / self.nb_actions_radar
        self.a_radar = np.random.choice(self.actions_radar, 1, p=self.str_radar)[0]
        #print(self.a_radar)
        self.iterCount=self.iterCount+1

        self.recent_history.append(self.a_radar)
        return self.a_radar

    def update_policy(self, a_jammer,jammer_strategy):
        self.T += 1
        self.str_update(a_jammer,jammer_strategy)



    def str_update(self, a_jammer,jammer_strategy):
        # Two steps: 1) Predict new str_radar 2) Update all the record matrix
        # k=0, stationary model and look-though model
        # str_radar_nohistory = self.hisdep.stationary(a_jammer, self.A_hat)
        str_radar_nohistory = np.ones(self.nb_actions_radar) / self.nb_actions_radar
        self.str_radar_mix[:, 0] = str_radar_nohistory
        self.str_radar_all[0] = str_radar_nohistory
        # k>0, str_radar is ar weighted sum
        # for i in np.arange(1, self.max_history):
        for i in np.arange(5, 6):
        # for i in np.arange(1, 2):
        # for i in np.arange(3, 3):
            # freq-dependent model
            # str_radar_freq = self.hisdep.history_dependent_freq(a_jammer, i, self.recent_history, self.A_hat)
            str_radar_freq = np.ones(self.nb_actions_radar) / self.nb_actions_radar
            # no-prior model
            # str_radar_nofreq = self.hisdep.history_dependent_nofreq(a_jammer, i, self.recent_history, self.A_hat)
            str_radar_nofreq = np.ones(self.nb_actions_radar) / self.nb_actions_radar
            # neural network
            if i >= 3:
                str_radar_nn = self.hisnn.neural_network_predict(a_jammer, i, self.recent_history, self.A_hat,jammer_strategy)
            else:
                str_radar_nn = np.ones(self.nb_actions_radar) / self.nb_actions_radar
            # Mixed strategy
            str_radar_total = np.array([str_radar_freq, str_radar_nofreq, str_radar_nn])
            str_radar_mix = np.dot(np.transpose(str_radar_total), self.select_model[i, :].reshape(-1))
            self.str_radar_mix[:, i] = str_radar_mix
            self.str_radar_all[i] = np.transpose(str_radar_total)

