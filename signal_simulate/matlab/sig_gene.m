function [radar_wave,jam_sig,len_sp]=sig_gene(radar_act,jammer_act,SweepBW,PRF,PW,NumPuls,NumSf,num_sp,ERP)

SweepBW=double(SweepBW);
PRF=double(PRF);
NumPuls=double(NumPuls);
RarPara = struct();
RarPara.PRF = PRF; % pulse repetion frequency 
RarPara.PW = PW; % pulse width (real pulse width: 100e-6)
RarPara.SweepBW = SweepBW; % sweep bandwidth width (total bandwidth for lfm) % Divided in three subpulses,each for 2MHz 
RarPara.NumPuls = NumPuls; % number of pulse    
RarPara.NumSf = NumSf;            %频率样本的数量



RarPara.NumSp = num_sp;        



%RarPara.NumSp表示雷达系统在距离上离散采样的点数或距离上的分辨率。
% 较大的NumSp可以提供更多的距离信息，从而实现更高的距离分辨率或者对多个目标的更准确探测
% 那是不是可以这么理解，在设置的该环境中，雷达仅仅发送了一个脉冲信号，该脉冲信号中采样了四个点进行目标距离的度量？
% RarPara.NumSp
Fs = 2*RarPara.SweepBW; %Sample Rate   
RarPara.Fs = Fs; 



waveform = phased.LinearFMWaveform('SampleRate',Fs,'PulseWidth',RarPara.PW,'PRF',RarPara.PRF,'SweepBandwidth',RarPara.SweepBW,'SweepDirection','Up',...
    'Envelope','Rectangular','OutputFormat','Pulses','NumPulses',RarPara.NumPuls,'SweepInterval','Positive'); 


jammer = barrageJammer('ERP',ERP,'SamplesPerFrame',waveform.NumPulses*waveform.SampleRate/waveform.PRF); % ERP: power % time for one pulse * sample rate 



wav_ori = waveform(); 
% figure;
% spectrogram(wav_ori, hamming(nsc),nov,nff, Fs, 'yaxis')  
% Generate Transmit Signal Referring to Radar Action 
[radar_wave, len_sp] = GenerateWave(radar_act, wav_ori, RarPara); 






jam_sig = jammer();
% % Synchronization with radar
num_stop = 1/2*length(jam_sig);
jam_sig(num_stop:end) = 0; 

jam_sig = GenerateJam(jammer_act, radar_act, jam_sig, Fs, len_sp);  
