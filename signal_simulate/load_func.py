import numpy as np
import matlab
import json
import matlab


def load_func(config_dir,eng,radar_act,jammer_act):
    radar_act=radar_act.reshape(1, -1)
    radar_act=matlab.double(radar_act+1.)
    jammer_act=matlab.double(jammer_act)
    config = json.load(open(config_dir))
    if 'Fc' in config:
        Fc = config['Fc']
    if 'PRF' in config:
        PRF = config['PRF']
    if 'PW' in config:
        PW = config['PW']
    if 'SweepBW' in config:
        SweepBW = config['SweepBW']
    if 'NumPuls' in config:
        NumPuls = config['NumPuls']
        NumPuls = matlab.double(NumPuls)
    if 'NumSf' in config:
        NumSf = config['NumSf']
    if 'Num_sp' in config:
        Num_sp = config['Num_sp']
    if 'PeakPower' in config:
        PeakPower = config['PeakPower']
    if 'Gain' in config:
        Gain = config['Gain']
        Gain = matlab.double(Gain)
    if 'ERP' in config:
        ERP = config['ERP']
    if 'MeanRCS' in config:
        MeanRCS = config['MeanRCS']
        MeanRCS = matlab.double(MeanRCS)
    if 'NoisePower' in config:
        NoisePower = config['NoisePower']
    if 'IfBackBaffled' in config:
        IfBackBaffled = config['IfBackBaffled']
    if 'InUseOutputPort' in config:
        InUseOutputPort = config['InUseOutputPort']
    if 'TwoWayPropagation_target' in config:
        TwoWayPropagation_target = config['TwoWayPropagation_target']
    if 'TwoWayPropagation_jammer' in config:
        TwoWayPropagation_jammer = config['TwoWayPropagation_jammer']
    if 'urasize' in config:
        urasize = config['urasize']
        urasize = np.array(urasize).reshape(1, -1)
        urasize = matlab.double(urasize)
    if 'Target_loc' in config:
        Target_loc = config['Target_loc']
        Target_loc = np.array(Target_loc).reshape(-1, 1)
        Target_loc = matlab.double(Target_loc)
    if 'Jammer_loc' in config:
        Jammer_loc = config['Jammer_loc']
        Jammer_loc = np.array(Jammer_loc).reshape(-1, 1)
        Jammer_loc = matlab.double(Jammer_loc)


    [reward,s_radar,s_echo,s_jammer]=eng.overall_func(radar_act,jammer_act,ERP,Fc,urasize,IfBackBaffled,PRF,PW,SweepBW,NumPuls,NumSf,PeakPower,Gain,InUseOutputPort,MeanRCS,TwoWayPropagation_target,TwoWayPropagation_jammer,NoisePower,Target_loc,Jammer_loc,Num_sp,nargout=4)

    return reward,s_radar,s_echo,s_jammer