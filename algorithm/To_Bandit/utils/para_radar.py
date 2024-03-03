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




