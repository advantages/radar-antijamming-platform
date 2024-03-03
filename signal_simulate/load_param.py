import numpy as np

import json



def load_param(config_dir):

    config = json.load(open(config_dir))
    if 'episode' in config:
        episode = config['episode']
    if 'NumSf' in config:
        num_sf = config['NumSf']
    if 'Num_sp' in config:
        num_sp = config['Num_sp']
    if 'jammer_type' in config:
        jammer_type = config['jammer_type']
    if 'history_len' in config:
        his_len = config['history_len']


    return episode,num_sf,num_sp,jammer_type,his_len