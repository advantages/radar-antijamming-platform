import numpy as np
from scipy.ndimage.filters import gaussian_filter1d
import scipy.stats as stats
import matplotlib
import pickle
import os


def hps_to_fstr(hps):
    return "_".join(["{}{}".format(hp, v) for hp, v in hps.items()])


def hps_to_tstr(hps):
    return " ".join(["{}={}".format(hp, v) for hp, v in hps.items()])


def singleplot(plt, iters, value, color, label, xscale, yscale, ylim_low, ylim_up, line):
    median = np.median(value, axis=0)
    plt.plot(iters, median, color=color, label=label, linestyle=line, linewidth=3.8)
    plt.grid(True)
    plt.ylim(ylim_low, ylim_up)
    plt.xlim(left=1)
    plt.xscale(xscale)
    plt.yscale(yscale)


def myplot(plt, iters, valuess, xscale, yscale, ylim_low, ylim_up):
    m = np.amin(valuess, axis=0)
    median = np.median(valuess, axis=0)
    M = np.amax(valuess, axis=0)
    plt.plot(iters, median)
    plt.fill_between(iters, m, M, alpha=0.3)
    plt.grid(True)
    plt.set_ylim(ylim_low, ylim_up)
    plt.set_xlim(left=1)
    plt.set_xscale(xscale)
    plt.set_yscale(yscale)

def mnplot(plt, iters, value, xscale, yscale, ylim_low, ylim_up):
    plt.scatter(iters, value)
    plt.grid(True)
    plt.set_ylim(ylim_low, ylim_up)
    plt.set_xlim(left=1)
    plt.set_xscale(xscale)
    plt.set_yscale(yscale)
    plt.set_aspect(1)

def multiplot(plt, iters, valuess, color, label, xscale, yscale, ylim_low, ylim_up, line):
    m = np.amin(valuess, axis=0)
    mean = np.mean(valuess, axis=0)
    M = np.amax(valuess, axis=0)
    plt.plot(iters, mean, color=color, label=label, linestyle=line, linewidth=5)
    # plt.fill_between(iters, m, M, alpha=0.3, color=color)
    plt.grid(True)
    plt.set_ylim(ylim_low, ylim_up)
    plt.set_xlim(left=1)
    plt.set_xscale(xscale)
    plt.set_yscale(yscale)

def multiplot_marker(plt, iters, valuess, color, label, xscale, yscale, ylim_low, ylim_up, marker,line):
    # m = np.amin(valuess, axis=0)
    median = np.mean(valuess, axis=0)
    # M = np.amax(valuess, axis=0)
    plt.plot(iters, median, color=color, label=label, marker=marker, markersize=15, markevery=50, linewidth=5, linestyle=line)
    # plt.fill_between(iters, m, M, alpha=0.3, color=color)
    plt.grid(True)
    plt.set_ylim(ylim_low, ylim_up)
    plt.set_xlim(left=1)
    plt.set_xscale(xscale)
    plt.set_yscale(yscale)

def multihist(plt, valuess, bin):
    plt.hist(valuess, bins=bin, edgecolor="black")
    # plt.scatter(iters, plot_y, color=color, label=label, s=12)
    # plt.grid(True)
    # plt.set_ylim(ylim_low, ylim_up)
    # plt.set_xlim(left=1)
    # plt.set_xscale(xscale)
    # plt.set_yscale(yscale)


def multiplot2(plt, iters, valuess, color, label, xscale, yscale, ylim_low, ylim_up):
    m = np.amin(valuess, axis=0)
    median = np.median(valuess, axis=0)
    M = np.amax(valuess, axis=0)
    plt.plot(iters, median, color=color, label=label)
    plt.fill_between(iters, m, M, alpha=0.3, color=color)
    plt.grid(True)
    plt.ylim(ylim_low, ylim_up)
    plt.xlim(left=1)
    plt.xscale(xscale)
    plt.yscale(yscale)


def compute_dist_dist(game, player, sigmai):
    if isinstance(sigmai, dict):
        sigmai = np.array(list(sigmai.values()))
    ne_sigmai = getattr(game, "sigma{}".format(player))
    return stats.entropy(ne_sigmai, sigmai)


def compute_kl(sigma0, sigma1):
    if isinstance(sigma0, dict):
        sigma0 = np.array(list(sigma0.values()))
    if isinstance(sigma1, dict):
        sigma1 = np.array(list(sigma1.values()))
    return stats.entropy(sigma0, sigma1)


def compute_players_utility(game, a0, a1):
    # u0 = {}
    # u1 = {}
    # for a_radar in actions:
    #     u0[a_radar] = game._u0_matrix[a_radar][a1]
    #     u1[a_radar] = -game._u0_matrix[a0][a_radar]
    u0 = game._u0_matrix[:, a1]
    u1 = -game._u0_matrix[a0, :]
    return u0, u1

def compute_players_utility_general(game, a0, a1):
    # u0 = {}
    # u1 = {}
    # for a_radar in actions:
    #     u0[a_radar] = game._u0_matrix[a_radar][a1]
    #     u1[a_radar] = -game._u0_matrix[a0][a_radar]
    u0 = game._u0_matrix[:, a1]
    u1 = game._u1_matrix[a0, :]
    return u0, u1


def compute_expected_gains(game, sigma0, sigma1):
    if isinstance(sigma0, dict):
        sigma0 = np.array(list(sigma0.values()))
    if isinstance(sigma1, dict):
        sigma1 = np.array(list(sigma1.values()))
    Ay = np.einsum("ij,j->i", game._u0_matrix, sigma1)
    E_gain_0 = np.einsum("i,i->", sigma0, Ay)
    return E_gain_0, (-1) * E_gain_0


def compute_expected_gains_vector(game, sigma0, sigma1):
    if isinstance(sigma0, dict):
        sigma0 = np.array(list(sigma0.values()))
    if isinstance(sigma1, dict):
        sigma1 = np.array(list(sigma1.values()))
    E_gain_vec_0 = np.einsum("ij,j->i", game._u0_matrix, sigma1)
    E_gain_vec_1 = (-1) * np.einsum("i,ij->j", sigma0, game._u0_matrix)
    return E_gain_vec_0, E_gain_vec_1


def save_game_settings(game_settings, save_path, file_name):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    file_path = os.path.join(save_path, file_name)
    # res = [game._u0_matrix, game.sigma0, game.v0, game.sigma1, game.v1]
    with open(file_path, "wb") as f:
        pickle.dump(game_settings, f)
    f.close()
    print("Saving game setting Done to {}".format(file_path))


def save_results(res, save_path, file_name):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    file_path = os.path.join(save_path, file_name)
    with open(file_path, "wb") as f:
        pickle.dump(res, f)
    f.close()
    print("Saving results Done to {}".format(file_path))


def load_data(load_path):
    with open(load_path, "rb") as f:
        data = pickle.load(f)
    f.close()
    return data

# For one pulse
def sigma2action_radar(sigma):
    action_0, action_1, action_2 = 0,0,0
    for a in range(len(sigma)):
        seq = Num2Act_Radar(a)
        for val in seq:
            if val == 0:
                action_0 += sigma[a]
            if val == 1:
                action_1 += sigma[a]
            if val == 2:
                action_2 += sigma[a]
    action = np.array([action_0, action_1, action_2])
    action = action/np.sum(action)
    action_prob = {a: action[a] for a in range(len(action))}

    return action_prob

def sigma2action_jammer(sigma):
    action_0, action_1, action_2 = 0,0,0
    for a in range(len(sigma)):
        seq = [a,a,a]
        for val in seq:
            if val == 0:
                action_0 += sigma[a]
            if val == 1:
                action_1 += sigma[a]
            if val == 2:
                action_2 += sigma[a]
    action = np.array([action_0, action_1, action_2])
    action = action/np.sum(action)
    action_prob = {a: action[a] for a in range(len(action))}

    return action_prob

# For two pulse
def sigma2action_radar_general(sigma):
    action_0, action_1, action_2 = 0,0,0
    for a in range(len(sigma)):
        seq = Num2Act_Radar_general(a)
        for val in seq:
            if val == 0:
                action_0 += sigma[a]
            if val == 1:
                action_1 += sigma[a]
            if val == 2:
                action_2 += sigma[a]
    action = np.array([action_0, action_1, action_2])
    action = action/np.sum(action)
    action_prob = {a: action[a] for a in range(len(action))}

    return action_prob

def sigma2action_jammer_general(sigma):
    action_0, action_1, action_2 = 0,0,0
    for a in range(len(sigma)):
        seq = Num2Act_Jammer_general(a)
        for val in seq:
            if val == 0:
                action_0 += sigma[a]
            if val == 1:
                action_1 += sigma[a]
            if val == 2:
                action_2 += sigma[a]
    action = np.array([action_0, action_1, action_2])
    action = action/np.sum(action)
    action_prob = {a: action[a] for a in range(len(action))}

    return action_prob



def Num2Act_Radar(a):
    z, z1 = a % 3, a // 3
    if z1 < 3:
        y, x = z1, 0
    else:
        y = z1 % 3
        x = a // 9
    return [x, y, z]

def Num2Act_Jammer(aj):
    return  [aj, aj, aj]

def Num2Act_Radar_general(a):
    m, n = a // 27, a % 27
    u, v, w = Num2Act_Radar(m)
    x, y, z = Num2Act_Radar(n)

    return [u, v, w, x, y, z]



def Num2Act_Jammer_general(aj):
    m, n = aj // 3, aj % 3

    return [m, m, m, n, n, n]


def subpulse_radar(sigma):
    subpulse = [Num2Act_Radar(i) for i in range(27)]
    sub0_0, sub0_1, sub0_2, sub1_0, sub1_1, sub1_2, sub2_0, sub2_1, sub2_2 = 0, 0, 0, 0, 0, 0, 0, 0, 0
    for i in range(27): # for each a_radar
        if subpulse[i][0] == 0:
            sub0_0 += sigma[i]
        elif subpulse[i][0] == 1:
            sub0_1 += sigma[i]
        elif subpulse[i][0] == 2:
            sub0_2 += sigma[i]
        if subpulse[i][1] == 0:
            sub1_0 += sigma[i]
        elif subpulse[i][1] == 1:
            sub1_1 += sigma[i]
        elif subpulse[i][1] == 2:
            sub1_2 += sigma[i]
        if subpulse[i][2] == 0:
            sub2_0 += sigma[i]
        elif subpulse[i][2] == 1:
            sub2_1 += sigma[i]
        elif subpulse[i][2] == 2:
            sub2_2 += sigma[i]

    sub0 = [sub0_0, sub0_1, sub0_2] # first subpulse
    sub1 = [sub1_0, sub1_1, sub1_2]
    sub2 = [sub2_0, sub2_1, sub2_2]

    return [sub0, sub1, sub2]












