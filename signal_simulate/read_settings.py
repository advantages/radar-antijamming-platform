import json
import numpy as np


config = json.load(open("./setting.json"))
print(config)
# print(type(np.array(config['Jammer_loc'])))
# print(np.array(config['Jammer_loc']).shape)