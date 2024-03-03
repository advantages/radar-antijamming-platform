import numpy as np
import argparse
import pickle
import os
import datetime
from utils.para_radar import para_radar
from utils.env_load import load_game
from utils.algo_load import load_algo
import sys

# Argparse part
parser = argparse.ArgumentParser()
parser.add_argument('--alg', type=str, required=True)
parser.add_argument('--iter', type=float, required=True)
parser.add_argument('--env_seed', type=int, required=True)
parser.add_argument('--alg_seed', type=int, required=True)
parser.add_argument('--jammer', type=str, required=True)
parser.add_argument('--cuda', type=int, required=True)
args = parser.parse_args()

##### PARAMETERS #####
num_sf = 4
num_sp = 4
max_history = 7
alg = args.alg
iteration = args.iter
env_seed = args.env_seed
alg_seed = args.alg_seed
jammertype = args.jammer
save_path = os.path.join("data_radar_test_1020/", jammertype, str(int(iteration)))
cuda_num=args.cuda
# Plot iters
iters = np.arange(1, iteration+1)
# plot_iters = np.arange(1, 10)
plot_iters = np.arange(1, 100)
iplot = 1
while plot_iters.max() < iteration:  # log-plot
    iplot *= 10
    # plot_iters = np.concatenate((plot_iters, iplot * np.arange(1, 10)))
    plot_iters = np.concatenate((plot_iters, iplot * np.arange(10, 100)))
plot_iters = plot_iters[plot_iters <= iteration]

# Environment
para = para_radar(alg, iteration, env_seed, alg_seed, num_sf, num_sp, max_history, cuda_num)
#para = para_radar(alg, iteration, env_seed, alg_seed, num_sf, num_sp, max_history)


game = load_game(para)


actions_01 = game.actions_radar
actions_02 = game.actions_jammer  # Not really use actions_02




S = np.zeros((game.nb_actions_radar, 1))
S1_value = 0
S2_value = 0
rewards = 0
Regret, Exp_Regret, Reward, Average_Reward = [], [], [], []
Radar_Strategy, Jammer_strategy = [], []
Radar_action, Jammer_action = [], []
Prob_hislen, Prob_model = [], []
algo1 = load_algo(actions_01, actions_02, para)
starttime = datetime.datetime.now()


for i in iters:
    a_radar = algo1.play()
    radar_strategy = algo1.str_radar
    rew, rew_vec, rew_exp, a_jammer, jammer_strategy = game.step_BLN(a_radar, jammertype)
    rewards += rew
    # Policy update
    if alg == 'BL':
        algo1.update_policy(a_jammer)  # Not really use a_radar
    elif alg == 'Exp3R':
        u_normal = rew-3
        algo1.update_policy(u_normal, a_jammer)
    elif alg == 'HedgeR':
        vec_normal = rew_vec - 3*np.ones(algo1.nb_action)
        algo1.update_policy(vec_normal, a_jammer)
    else:
        algo1.update_policy(rew, a_jammer)
    T = algo1.T
    average_reward = 1 / T * rewards
    # Calculate Expected Regret
    S1_value = S1_value + rew_exp[a_radar]
    exp_regret = np.max(rew_exp) - 1 / T * S1_value
    # Calculate Average Regret
    s_vec = rew_vec-rew_vec[a_radar]
    S2_value = S2_value + s_vec.max()
    regret = 1 / T * S2_value
    radar_action = a_radar
    jammer_action = a_jammer
    if alg == 'BL':
        prob_hislen = algo1.select_history
        hislen = int(jammertype[-1])
        prob_model = algo1.select_model[hislen, :]

    # if T in plot_iters:
    if (T < 100) or ((T % 10 == 0) and (T in plot_iters)):
        Regret.append(np.float32(regret))
        Exp_Regret.append(np.float32(exp_regret))
        Reward.append(np.float32(rew))
        Average_Reward.append(np.float32(average_reward))
        Radar_Strategy.append(np.float32(radar_strategy))
        Jammer_strategy.append(np.float32(jammer_strategy))
        Radar_action.append(radar_action)
        Jammer_action.append(jammer_action)
        if alg == 'BL':
            Prob_hislen.append(prob_hislen)
            Prob_model.append(prob_model)
        print(alg, '-', T, '-', rew)
        # print(alg, '-', T, '-', algo1.select_model[2, 0])
        # print(a_radar, '-', a_jammer)


# Data
res_dict = {
    "Regret": Regret,
    "Exp_Regret": Exp_Regret,
    "Reward": Reward, 
    "Ave_Reward": Average_Reward,
    "plot_iter": plot_iters,
    "action_radar": Radar_action,
    "action_jammer": Jammer_action,
    "strategy_radar": Radar_Strategy,
    "strategy_jammer": Jammer_strategy,
    'prob_his': Prob_hislen,
    'prob_model': Prob_model,
}

# Time
endtime = datetime.datetime.now()
seconds = (endtime-starttime).seconds
hours = int(seconds/3600*1000)/1000

##### SAVE DATA #####
# save path
filename ="{}_{}_{}_{}_{}.data".format(alg, str(int(iteration)), str(env_seed), str(alg_seed), str(hours))
file_path = os.path.join(save_path, filename)
if not os.path.exists(save_path):
    os.makedirs(save_path)
if (os.path.exists(file_path)):
    os.remove(file_path)
# save file
with open(file_path, "wb") as fp:
    pickle.dump(res_dict, fp)
fp.close()
print("Saving results Done to {}".format(file_path))


