from algorithms import (
    Bandit_Level,
)

def load_algo(action01, action02, para):
    algo = load_koa_algo(action01, action02, para)

    return algo

def load_koa_algo(actions, actions_oppo, args):

    if args.algo == "Bandit_Level":
        return Bandit_Level(actions, actions_oppo, args.alg_seed, args.num_sf, args.num_sp, args.max_history,args.cuda_num)
    else:
        raise NotImplementedError("Not KOA algorithms")

