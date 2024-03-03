import numpy as np
import scipy.stats as stats
from .utils import sigma2action_radar, sigma2action_jammer
from .utils import sigma2action_radar_general, sigma2action_jammer_general



def load_sigma(algo, para, env, dim):
    avg_sigma = algo.asigmai
    last_sigma = algo.sigmai
    if env == 'radar' or env == 'inr':
        # Turn actions' probability simplex to subpulses' probability simplex
        # Pulse is probability simplex on all the subpulses (in our env, just three pulses)
        if dim == "01":
            if para.algo == "Fix":
                avg_sigma = algo.apulse
                last_sigma = algo.pulse
            else:
                avg_sigma = sigma2action_radar(avg_sigma)
                last_sigma = sigma2action_radar(last_sigma)
        elif dim == "02":
            if para.algo == "FixGeneral":
                avg_sigma = algo.apulse
                last_sigma = algo.pulse
            else:
                avg_sigma = sigma2action_jammer_general(avg_sigma)
                last_sigma = sigma2action_jammer_general(last_sigma)

    return avg_sigma, last_sigma


def compute_regret(player, game, S, a1, a2, T):
    u1 = game._u0_matrix[:, a2]
    u2 = -game._u0_matrix[a1, :]
    if player == 0:
        S = S + u1 - u1[a1]
    else:
        S = S + u2 - u2[a2]
    Regret = 1 / T * S.max()

    return Regret, S

def compute_regret2(game, S1, S2, a1, a2, T):
    u1 = game._u0_matrix[:, a2]
    u2 = -game._u0_matrix[a1, :]
    S1 = S1 + u1 - u1[a1]
    S2 = S2 + u2 - u2[a2]
    Regret_1 = 1 / T * S1.max()
    Regret_2 = 1 / T * S2.max()

    return Regret_1, Regret_2, S1, S2


def compute_expected_gains(game, sigma0, sigma1):
    if isinstance(sigma0, dict):
        sigma0 = np.array(list(sigma0.values()))
    if isinstance(sigma1, dict):
        sigma1 = np.array(list(sigma1.values()))
    Ay = np.einsum("ij,j->i", game._u0_matrix, sigma1)
    E_gain_0 = np.einsum("i,i->", sigma0, Ay)
    return E_gain_0, (-1) * E_gain_0, sigma0, sigma1


def compute_expected_gains_vector(game, sigma0, sigma1):
    if isinstance(sigma0, dict):
        sigma0 = np.array(list(sigma0.values()))
    if isinstance(sigma1, dict):
        sigma1 = np.array(list(sigma1.values()))
    E_gain_vec_0 = np.einsum("ij,j->i", game._u0_matrix, sigma1)
    E_gain_vec_1 = (-1) * np.einsum("i,ij->j", sigma0, game._u0_matrix)
    return E_gain_vec_0, E_gain_vec_1


def compute_expect_regret(game, avg_sigma0, avg_sigma1, last_sigma0, last_sigma1, sum_last, T):
    E_gain_0, E_gain_1, sigma0, sigma1 = compute_expected_gains(game, last_sigma0, last_sigma1)
    E_gain_vec_0, E_gain_vec_1 = compute_expected_gains_vector(game, avg_sigma0, avg_sigma1)
    sum_last = sum_last + E_gain_0
    E_regret_0 = np.max(E_gain_vec_0) - 1 / T * sum_last
    E_regret_1 = np.max(E_gain_vec_1) + 1 /T * sum_last
    dual_gap = E_regret_0 + E_regret_1
    return E_regret_0, E_regret_1, dual_gap, sum_last, sigma0, sigma1



def compute_kl(game, player, sigmai):
    if isinstance(sigmai, dict):
        sigmai = np.array(list(sigmai.values()))
    ne_sigmai = getattr(game, "sigma{}".format(player)) # nash strategy
    return stats.entropy(ne_sigmai, sigmai)


def load_analysis(player, algo, a1, a2, T, dim, env, para, game, S):
    avg_sigma, last_sigma = load_sigma(algo, para, env, dim)
    Regret, S = compute_regret(player, game, S, a1, a2, T)
    KL = compute_kl(game, player, avg_sigma)
    # KL = [0.5,0.2,0.1]

    return Regret, KL, S, last_sigma

def load_analysis_full(algo1, algo2, a1, a2, T, dim, env, para1, game, S1, S2, sum_last):
    avg_sigma_1, last_sigma_1 = load_sigma(algo1, para1, env, dim)
    avg_sigma_2, last_sigma_2 = load_sigma(algo2, para1, env, dim)
    Regret_1, Regret_2, S1, S2 = compute_regret2(game, S1, S2, a1, a2, T)
    E_regret_1, E_regret_2, dual_gap, sum_last, sigma0, sigma1 = compute_expect_regret(game, avg_sigma_1, avg_sigma_2, last_sigma_1, last_sigma_2, sum_last, T)
    KL1 = compute_kl(game, 0, avg_sigma_1)
    KL2 = compute_kl(game, 1, avg_sigma_2)

    return Regret_1, Regret_2, E_regret_1, E_regret_2, KL1, KL2, dual_gap, S1, S2, sum_last, sigma0, sigma1

def load_analysis_radar(algo1, algo2, a1, a2, T, game, S1, S2, sum_last):
    avg_sigma_1 = algo1.asigmai
    last_sigma_1 = algo1.sigmai
    avg_sigma_2 = algo2.asigmai
    last_sigma_2 = algo2.sigmai
    Regret_1, Regret_2, S1, S2 = compute_regret2(game, S1, S2, a1, a2, T)
    E_regret_1, E_regret_2, dual_gap, sum_last, sigma0, sigma1 = compute_expect_regret(game, avg_sigma_1, avg_sigma_2, last_sigma_1, last_sigma_2, sum_last, T)

    return Regret_1, Regret_2, E_regret_1, E_regret_2, dual_gap, S1, S2, sum_last, sigma0, sigma1

def analysis_hedge(algo, game, a1, a2, sum1, T):
    post_mean = algo.post_mean[a1, a2]
    u1 = algo.A_hat[:, a2]
    # u1 = game._u0_matrix[:, a2]
    diff = u1 - post_mean
    modi_diff = np.maximum(diff, 0)
    # modi_diff = diff
    sum1 = sum1 + modi_diff
    Modify_Regret = 1 / T * sum1.max()

    return Modify_Regret, sum1

def analysis_adv(a1, sum2, T, u1):
    diff = u1 - u1[a1]
    modi_diff = np.maximum(diff, 0)
    # modi_diff = diff
    sum2 = sum2 + modi_diff
    Adv_Regret = 1 / T * sum2.max()

    return Adv_Regret, sum2



















