import numpy as np
import matlab
import sig_gene
import collector_gene
import reward_gene
import time



Fc=10e9
PRF=1e3
PW=60e-6
SweepBW=20e6
NumPuls=1
NumSf=8
Num_sp=3
ERP=1e6




jammer_action_v = np.array([1.0, 1. ]).reshape(1, -1)
jammer_action_v = matlab.double(jammer_action_v)


radar_action_v = np.array([3.0,3.0,3.0]).reshape(1, -1)
radar_action_v = matlab.double(radar_action_v)


env1=sig_gene.initialize()
env2=collector_gene.initialize()
env3=reward_gene.initialize()





################################## Param2 ################################



Target_loc=np.array([10000.0,5000.0,0.0]).reshape(-1,1)
Jammer_loc=np.array([10000.0,5000.0,0.0]).reshape(-1,1)
Target_loc=matlab.double(Target_loc)
Jammer_loc=matlab.double(Jammer_loc)


PeakPower=1e9

Gain=60.0
Gain=matlab.double(Gain)


InUseOutputPort=True

urasize=np.array([10.0,20.0]).reshape(1,-1)
urasize=matlab.double(urasize)

IfBackBaffled=True

MeanRCS=10.0
MeanRCS=matlab.double(MeanRCS)

TwoWayPropagation_target=True
TwoWayPropagation_jammer=False

NoisePower=1e-13



start_time = time.time()


[radar_wave,jam_sig,len_sp]=env1.sig_gene(radar_action_v,jammer_action_v,SweepBW,PRF,PW,NumPuls,NumSf,Num_sp,ERP,nargout=3)








[radar_sig,echo_sig,jammer_sig]=env2.collector_gene(radar_wave,jam_sig,Target_loc,Jammer_loc,SweepBW,PeakPower,Gain,InUseOutputPort,urasize,IfBackBaffled,Fc,MeanRCS,TwoWayPropagation_target,TwoWayPropagation_jammer,NoisePower,nargout=3)







[reward,fil_nojam,fil_jam]=env3.reward_gene(radar_sig,jammer_sig,echo_sig,SweepBW,PRF,len_sp,radar_action_v,radar_wave,nargout=3)

end_time = time.time()
print("init time", end_time - start_time, "秒")


print(reward)



env1.quit()
env2.quit()
env3.quit()

