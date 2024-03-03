import sys
sys.path.append('../../jammer_environment')
from jammer_env import Jammer_single


def load_game(args):
    if args.game == "Radar":
        game = Jammer_single(args.num_sf, args.num_sp, args.env_seed)
    else:
        raise NotImplementedError("Not implemented.")
    return game