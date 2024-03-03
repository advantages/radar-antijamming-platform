class para_choose:
    def __init__(self, algo, iteration, env):
        # environment
        if env == "radar":
            self.game = "RadarEnv01"
        elif env == "radarsub":
            self.game = "RadarEnvSubpulse"
        elif env == "inr":
            self.game = "RadarINR"
        elif env == "bNFG":
            self.game = "bNFG"
        elif env == "bNFG_w10":
            self.game = "bNFG_worst_10"
        elif env == "bNFG_wl":
            self.game = "bNFG_worst_large"
            self.nb_actions = 20
        elif env == "bNFG_t":
            self.game = "bNFG_truncnorm"
            self.nb_actions = 20
        elif env == "bNFG_rps":
            self.game = "bNFG_rps_like"
            self.nb_actions = 20
        elif env == "mrps":
            self.game = "mRPS"
        elif env == "radarbandit":
            self.game = "RadarBandit"
        elif env == "radarlevel":
            self.game = "RadarLevel"
        else:
            pass

        # Choose algo&Para
        if algo == "Exp3":
            self.algo = "Exp3Auto"
            self.iters = iteration
            self.eta_scheme = "constant"  # "constant" or "decreasing"
            # self.avg_scheme = "weighted"  # "weighted" or None
            self.avg_scheme = None
            self.delta = 0

        if algo == "Exp3R":
            self.algo = "Exp3_radar"
            self.iters = iteration

        if algo == "HedgeR":
            self.algo = "Hedge_radar"
            self.iters = iteration

        elif algo == "Exp3p":
            self.algo = "Exp3P"
            self.iters = iteration
            self.eta = 0.01
            self.delta = 0
            self.gamma = 0
            self.beta = 0

        elif algo == "ExpEx":
            self.algo = "ExpExtend"
            self.iters = iteration

        elif algo == "TSB":
            self.algo = "TS_Bandit"
            self.prior_mean = 0
            self.prior_var = 1

        elif algo == "OTSB":
            self.algo = "OTS_Bandit"
            self.prior_mean = 0
            self.prior_var = 1

        elif algo == "UCBB":
            self.algo = "UCB_Bandit"
            self.iters = iteration

        elif algo == "Prac":
            self.algo = "Practical_Radar"
            self.type = 1


        elif algo == "TS":
            self.algo = "TS_EW_KOA"
            self.iters = iteration
            self.alg_prior = "Gauss"
            self.eta_scheme = "constant"
            # self.eta_scheme = "decreasing"
            self.prior_mean = 0
            self.prior_var = 1


        elif algo == "TSE":
            self.algo = "TS_Empirical"
            self.iters = iteration
            self.alg_prior = "Gauss"
            self.eta_scheme = "constant"
            # self.eta_scheme = "decreasing"

        elif algo == "TSFE":
            self.algo = "TS_FiniteEmpirical"
            self.iters = iteration
            self.alg_prior = "Gauss"
            self.eta_scheme = "constant"
            # self.eta_scheme = "decreasing"
            self.s = 100

        elif algo == "TSO":
            self.algo = "TS_OppoModel"
            self.alg_prior = "Gauss"

        elif algo == "FixSt":
            self.algo = "Fix_Strategy"
            self.iters = iteration

        elif algo == "OCB":
            self.algo = "OCB_EW_KOA"
            self.iters = iteration

        elif algo == "OCBF":
            self.algo = "OCBF"
            self.iters = iteration

        elif algo == "Fix":
            self.algo = "Fix"
            self.iters = iteration

        elif algo == "FixGeneral":
            self.algo = "FixGeneral"
            self.iters = iteration

        elif algo == "INR":
            self.algo = "TS_INR"
            self.iters =iteration

        elif algo == "EWF":
            self.algo = "EWF"
            self.iters = iteration

        elif algo == "OEWF":
            self.algo = "OEWF"
            self.iters = iteration
            self.eta = 1e-2

        elif algo == "RM":
            self.algo = "RM"
            self.iters = iteration
            self.delta = 0

        elif algo == "TRM":
            self.algo = "TS_RM"
            self.iters = iteration
            self.alg_prior = "Gauss"
            self.prior_mean = 0.5
            self.prior_var = 0.1
            self.prior_var0 = 0.1

        elif algo == "TRMP":
            self.algo = "TS_RMP"
            self.iters = iteration
            self.alg_prior = "Gauss"
            self.prior_mean = 0.5
            self.prior_var = 0.1
            self.prior_var0 = 0.1

        elif algo == "URM":
            self.algo = "OCB_RM"
            self.iters = iteration

        elif algo == "TSC":
            self.algo = "TS_Clip"
            self.iters = iteration
            self.alg_prior = "Gauss"
            self.eta_scheme = "constant"
            # self.eta_scheme = "decreasing"
            self.prior_mean = 0.5
            self.prior_var = 0.1

        elif algo == "OTSC":
            self.algo = "Optimistic_TS"
            self.iters = iteration
            self.alg_prior = "Gauss"
            self.eta_scheme = "constant"
            # self.eta_scheme = "decreasing"
            self.prior_mean = 0
            self.prior_var = 1

        elif algo == "OTSF":
            self.algo = "OTSF"
            self.iters = iteration
            self.alg_prior = "Gauss"
            self.eta_scheme = "constant"
            # self.eta_scheme = "decreasing"
            self.prior_mean = 0
            self.prior_var = 1

        elif algo == "OTRM":
            self.algo = "OTS_RM"
            self.iters = iteration
            self.alg_prior = "Gauss"
            self.prior_mean = 0.5
            self.prior_var = 0.1
            self.prior_var0 = 0.1
            self.M = 10

        elif algo == "BUCB":
            self.algo = "BayesUCB"
            self.iters = iteration
            self.prior_mean = 0
            self.prior_var = 1
            self.beta = 1

        elif algo == "RMF":
            self.algo = "RMF"
            self.iters = iteration

        elif algo == "BL":
            self.algo = "Bandit_Level"
            self.prior_mean = 0
            self.prior_var = 1

        elif algo == "Rand":
            self.algo = "Random_Strategy"
            self.iters = iteration


class para_choose_dependence:
    def __init__(self, algo, iteration, nb_actions, env, seed, beta):
        # Environment
        if env == "bNFG_unif":
            self.game = "bNFG_Unif_Bern"
            self.nb_actions = nb_actions
            self.instance_seed = seed
            self.delta = 0

        elif env == "inr":
            self.game = "RadarINR"
        elif env == "adv":
            self.game = "adv"
            self.nb_actions = nb_actions

        # Algorithms
        if algo == "TS":
            self.algo = "TS_EW_KOA"
            self.iters = iteration
            self.alg_prior = "Gauss"
            self.eta_scheme = "constant"
            # self.eta_scheme = "decreasing"
            self.prior_mean = 0
            self.prior_var = 1

        elif algo == "Exp3":
            self.algo = "Exp3Auto"
            self.iters = iteration
            self.eta_scheme = "constant"  # "constant" or "decreasing"
            # self.avg_scheme = "weighted"  # "weighted" or None
            self.avg_scheme = None
            self.delta = 0

        elif algo == "OTS":
            self.algo = "TS_OEW_KOA"
            self.iters = iteration
            self.alg_prior = "Gauss"
            self.eta_scheme = "constant"
            # self.eta_scheme = "decreasing"

        elif algo == "RM":
            self.algo = "RM"
            self.iters = iteration
            self.delta = 0

        elif algo == "OCB":
            self.algo = "OCB_EW_KOA"
            self.iters = iteration
            self.beta = beta

        elif algo == "TRM":
            self.algo = "TS_RM"
            self.iters = iteration
            self.alg_prior = "Gauss"
            self.prior_mean = 0
            self.prior_var = 1
            self.prior_var0 = 0.1

        elif algo == "TRMP":
            self.algo = "TS_RMP"
            self.iters = iteration
            self.alg_prior = "Gauss"
            self.prior_mean = 0
            self.prior_var = 1
            self.prior_var0 = 0.1

        elif algo == "OTRM":
            self.algo = "OTS_RM"
            self.iters = iteration
            self.alg_prior = "Gauss"
            self.prior_mean = 0.5
            self.prior_var = 0.1
            self.prior_var0 = 0.1
            self.M = int(beta)

        elif algo == "OTRMP":
            self.algo = "OTS_RMP"
            self.iters = iteration
            self.alg_prior = "Gauss"
            self.prior_mean = 0
            self.prior_var = 1
            self.prior_var0 = 0.1
            self.M = 10

        elif algo == "URM":
            self.algo = "OCB_RM"
            self.iters = iteration
            self.beta = beta

        elif algo == "INR":
            self.algo = "TS_INR"
            self.iters =iteration

        elif algo == "Fix":
            self.algo = "Fix"
            self.iters = iteration

        elif algo == "EWF":
            self.algo = "EWF"
            self.iters = iteration

        elif algo == "PM":
            self.algo = "TS_Mean"
            self.iters = iteration
            self.alg_prior = "Gauss"
            self.eta_scheme = "constant"
            # self.eta_scheme = "decreasing"
            self.prior_mean = 0
            self.prior_var = 1

        elif algo == "TSC":
            self.algo = "TS_Clip"
            self.iters = iteration
            self.alg_prior = "Gauss"
            self.eta_scheme = "constant"
            self.prior_mean = 0
            self.prior_var = 1

        elif algo == "MTS":
            self.algo = "Max_TSC"
            self.iters = iteration
            self.alg_prior = "Gauss"
            self.eta_scheme = "constant"
            # self.eta_scheme = "decreasing"
            self.prior_mean = 0
            self.prior_var = 1

        elif algo == "MTRM":
            self.algo = "Max_TSRM"
            self.iters = iteration
            self.alg_prior = "Gauss"
            self.eta_scheme = "constant"
            # self.eta_scheme = "decreasing"
            self.prior_mean = 0
            self.prior_var = 1

        elif algo == "OMTS":
            self.algo = "Max_OTSC"
            self.iters = iteration
            self.alg_prior = "Gauss"
            self.eta_scheme = "constant"
            # self.eta_scheme = "decreasing"
            self.prior_mean = 0
            self.prior_var = 1

        elif algo == "OMTRM":
            self.algo = "Max_OTSRM"
            self.iters = iteration
            self.alg_prior = "Gauss"
            self.eta_scheme = "constant"
            # self.eta_scheme = "decreasing"
            self.prior_mean = 0
            self.prior_var = 1

        elif algo == "EM":
            self.algo = "Emprical_Mean"
            self.iters = iteration

        elif algo == "OTSC":
            self.algo = "Optimistic_TS"
            self.iters = iteration
            self.alg_prior = "Gauss"
            self.eta_scheme = "constant"
            # self.eta_scheme = "decreasing"
            self.prior_mean = 0.5
            self.prior_var = 0.1

        elif algo == "OTSOP":
            self.algo = "OTSOP"
            self.iters = iteration
            self.alg_prior = "Gauss"
            self.eta_scheme = "constant"
            # self.eta_scheme = "decreasing"
            self.prior_mean = 0
            self.prior_var = 1

        elif algo == "BR":
            self.algo = "Best_Response"

        elif algo == "BUCB":
            self.algo = "BayesUCB"
            self.iters = iteration
            self.prior_mean = 0
            self.prior_var = 1
            self.beta = 1

        elif algo == "RMF":
            self.algo = "RMF"
            self.iters = iteration

        elif algo == "RMPF":
            self.algo = "RMPF"
            self.iters = iteration

        elif algo == "MEW":
            self.algo = "Mean_Hedge"
            self.iters = iteration

        elif algo == "MRM":
            self.algo = "Mean_RM"
            self.iters = iteration

        elif algo == "FixSt":
            self.algo = "Fix_Strategy"
            self.iters = iteration

        elif algo == "JumpSt":
            self.algo = "Jump_Strategy"
            self.iters = iteration
            self.seed = seed



class para_counter:
    def __init__(self, algo, iteration, nb_actions, env, seed, delta, sigma, M):
        # Environment
        if env == "bNFG_unif":
            self.game = "bNFG_Unif_Bern"
            self.nb_actions = nb_actions
            self.instance_seed = seed
            self.delta = delta

        if algo == "TRM":
            self.algo = "TS_RM"
            self.iters = iteration
            self.alg_prior = "Gauss"
            self.prior_mean = 0
            self.prior_var = 1
            self.prior_var0 = sigma

        elif algo == "TRMP":
            self.algo = "TS_RMP"
            self.iters = iteration
            self.alg_prior = "Gauss"
            self.prior_mean = 0
            self.prior_var = 1
            self.prior_var0 = sigma

        elif algo == "MRM":
            self.algo = "Mean_RM"
            self.iters = iteration

        elif algo == "OTRM":
            self.algo = "OTS_RM"
            self.iters = iteration
            self.alg_prior = "Gauss"
            self.prior_mean = 0
            self.prior_var = 1
            self.prior_var0 = sigma
            self.M = M

        elif algo == "OTRMP":
            self.algo = "OTS_RMP"
            self.iters = iteration
            self.alg_prior = "Gauss"
            self.prior_mean = 0
            self.prior_var = 1
            self.prior_var0 = sigma
            self.M = M

        elif algo == "BR":
            self.algo = "Best_Response"

class para_choose_oppo:
    def __init__(self, algo, iteration, nb_actions, nb_oppo_actions, env, seed, stepsize):
        # Environment
        if env == "bNFG_oppo":
            self.game = "bNFG_Unif_Bern2"
            self.nb_actions = nb_actions
            self.nb_oppo_actions = nb_oppo_actions
            self.instance_seed = seed

        if algo == "OTSOP":
            self.algo = "OTSOP"
            self.iters = iteration
            self.alg_prior = "Gauss"
            self.eta_scheme = "constant"
            # self.eta_scheme = "decreasing"
            self.prior_mean = 0
            self.prior_var = 1
            self.stepsize = stepsize

        elif algo == "OTSC":
            self.algo = "Optimistic_TS"
            self.iters = iteration
            self.alg_prior = "Gauss"
            self.eta_scheme = "constant"
            # self.eta_scheme = "decreasing"
            self.prior_mean = 0
            self.prior_var = 1

        elif algo == "OTRM":
            self.algo = "OTS_RM"
            self.iters = iteration
            self.alg_prior = "Gauss"
            self.prior_mean = 0
            self.prior_var = 1
            self.prior_var0 = 1
            self.M = 10

        elif algo == "OTRMOP":
            self.algo = "OTRMOP"
            self.iters = iteration
            self.alg_prior = "Gauss"
            self.prior_mean = 0
            self.prior_var = 1
            self.prior_var0 = 1
            self.M = 10
            self.stepsize = stepsize

        elif algo == "BR":
            self.algo = "Best_Response"


class para_radar:
    def __init__(self, algo, iteration, env_seed, alg_seed, num_sf, num_sp, max_history,cuda_num):
        self.game = 'Radar'
        self.num_sf = num_sf
        self.num_sp = num_sp
        self.env_seed = env_seed


        if algo == 'BL':
            self.algo = 'Bandit_Level'
            self.iters = iteration
            self.max_history = max_history
            self.alg_seed = alg_seed
            self.cuda_num=cuda_num


        if algo == 'NN':
            self.algo = 'Neural_Network'
            self.iters = iteration
            self.max_history = max_history
            self.alg_seed = alg_seed
            self.cuda_num=cuda_num

        if algo == 'Exp3R':
            self.algo = 'Exp3_radar'
            self.iters = iteration

        if algo == 'HedgeR':
            self.algo = 'Hedge_radar'
            self.iters = iteration



