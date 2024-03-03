import numpy as np
import matlab
import overall_func
# import time


Fc=10e9
urasize=np.array([10.0,20.0]).reshape(1,-1)
urasize=matlab.double(urasize)
IfBackBaffled=True
PRF=1e3
PW=60e-6
SweepBW=20e6
NumPuls=1
NumSf=8
Num_sp=3
PeakPower=1e9
Gain=60.0
Gain=matlab.double(Gain)
InUseOutputPort=True
ERP=1e6
MeanRCS=10.0
MeanRCS=matlab.double(MeanRCS)
TwoWayPropagation_target=True
TwoWayPropagation_jammer=False
NoisePower=1e-13
Target_loc=np.array([10000.0,5000.0,0.0]).reshape(-1,1)
Jammer_loc=np.array([10000.0,5000.0,0.0]).reshape(-1,1)
Target_loc=matlab.double(Target_loc)
Jammer_loc=matlab.double(Jammer_loc)




jammer_action_v = np.array([1.0, 1. ]).reshape(1, -1)
jammer_action_v = matlab.double(jammer_action_v)


radar_action_v = np.array([3.0,3.0,3.0]).reshape(1, -1)
radar_action_v = matlab.double(radar_action_v)


env=overall_func.initialize()

start_time = time.time()

[reward,s_radar,s_echo,s_jammer]=env.overall_func(radar_action_v,jammer_action_v,ERP,Fc,urasize,IfBackBaffled,PRF,PW,SweepBW,NumPuls,NumSf,PeakPower,Gain,InUseOutputPort,MeanRCS,TwoWayPropagation_target,TwoWayPropagation_jammer,NoisePower,Target_loc,Jammer_loc,Num_sp,nargout=4)


end_time = time.time()
print("init time", end_time - start_time, "秒")


s_radar=np.array(s_radar)
s_echo=np.array(s_echo)
s_jammer=np.array(s_jammer)
print(reward)

env.quit()