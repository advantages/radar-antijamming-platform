import matlab
import matlab.engine
import numpy as np
import scipy
import argparse


parser = argparse.ArgumentParser(description='Generate reward matrix')

parser.add_argument('-sp', '--numsp',type=int, default=3,help='num of subpuls')
parser.add_argument('-sf', '--numsf',type=int, default=3,help='num of subfreq')
args = parser.parse_args()

num_sp=args.numsp
num_sf=args.numsf

eng = matlab.engine.start_matlab()

Mat=np.zeros((num_sf.__pow__(num_sp),num_sf))
def Num2Act_Radar(a):
    x = []
    for _ in range(num_sp):
        x.append(a % num_sf)  # Get the remainder of y when divided by 10
        a = a // num_sf  # Divide y by 10 to move to the next digit
    x.reverse()  # Reverse the order of elements in x
    return x


for i in range(num_sf.__pow__(num_sp)):
    for j in range(num_sf):
        radar_action_v=Num2Act_Radar(i)
        radar_action_v = np.array(radar_action_v)
        radar_action_v = radar_action_v.reshape(1, -1)
        radar_action_v = matlab.double(radar_action_v + 1.)

        jammer_action_v = np.array([1.0, j + 1.]).reshape(1, -1)
        jammer_action_v = matlab.double(jammer_action_v)
        reward=eng.gene_single_mat_val(radar_action_v,jammer_action_v,num_sf,num_sp)
        print(radar_action_v,jammer_action_v)
        print(reward)
        Mat[i,j]=reward

print(Mat)

scipy.io.savemat('reward_{}_{}.mat'.format(int(num_sf),int(num_sp)),Mat)
eng.quit()