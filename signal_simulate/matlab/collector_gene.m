function [radar_sig,echo_sig,jammer_sig]=collector_gene(radar_wave,jam_sig,Target_loc,Jammer_loc,SweepBW,PeakPower,Gain,InUseOutputPort,urasize,IfBackBaffled,Fc,MeanRCS,TwoWayPropagation_target,TwoWayPropagation_jammer,NoisePower)



element = phased.IsotropicAntennaElement('FrequencyRange', [0.75*Fc 1.25*Fc]);
% [0.75*Fc 1.25*Fc]

% creates an antenna element with an isotropic response pattern 
% urasize = [10 20]; % Planner array 
taper = taylorwin(urasize(1))*taylorwin(urasize(2))'; % tapper weights

% taper
URA = phased.URA('Size',urasize, 'Element',element,'Taper',taper);  % URA 
URA.Element.BackBaffled = IfBackBaffled; % no back 
lambda = physconst('lightspeed')/Fc; % wave length 
% lambda
URA.ElementSpacing = [lambda/2 lambda/2]; % ElementSpacing 



Fs = 2*SweepBW;


targetloc = Target_loc;   
jammerloc = Jammer_loc;    
[~,tgtang] = rangeangle(targetloc); %  path length and direction angles  
[~,jamang] = rangeangle(jammerloc); 



transmitter = phased.Transmitter('PeakPower',PeakPower,'Gain',Gain,'InUseOutputPort',InUseOutputPort); % Output power: 5e2*1e6=5e8 
radiator = phased.Radiator('Sensor',URA,'OperatingFrequency',Fc);  %  coherent sum of the delayed radiated fields from all elements  


target = phased.RadarTarget('Model','Nonfluctuating','MeanRCS',MeanRCS,'OperatingFrequency',Fc); % 10 m^2 
targetchannel = phased.FreeSpace('TwoWayPropagation',TwoWayPropagation_target,'SampleRate',Fs,'OperatingFrequency', Fc);    
jammerchannel = phased.FreeSpace('TwoWayPropagation',TwoWayPropagation_jammer,'SampleRate',Fs,'OperatingFrequency', Fc);   



collector = phased.Collector('Sensor',URA,'OperatingFrequency',Fc); 
% amplifier = phased.ReceiverPreamp('EnableInputPort',false); % all noise 
% amplifier = phased.ReceiverPreamp('EnableInputPort',true,NoiseFigure=1, SampleRate=Fs); % all noise  
amplifier = phased.ReceiverPreamp('NoiseMethod','Noise power','NoisePower',NoisePower); % all noise 



[wav,txstatus] = transmitter(radar_wave);
% Radiate pulse toward the target
wav = radiator(wav,tgtang); 
% Propagate pulse toward the target, location + velocity
wav = targetchannel(wav,[0;0;0],targetloc,[0;0;0],[0;0;0]); % loc1, loc2, vel1, vel2  
% s = wav(1:RarPara.NumPuls,1); 
% spectrogram(s, hamming(nsc),nov,nff, Fs, 'yaxis')  

% Reflect it off the target 
wav = target(wav);  
% plot(abs(wav))
% Collect the echo 
wav = collector(wav,tgtang);






jam_sig = jammerchannel(jam_sig,jammerloc,[0;0;0],[0;0;0],[0;0;0]);   
jam_sig = collector(jam_sig,jamang); 





%% Received Mixed Signal
% radar_sig = amplifier(wav, ~txstatus); 
% echo_sig = amplifier(wav + jam_sig, ~txstatus);  
% jammer_sig = amplifier(jam_sig, ~txstatus); 
radar_sig = amplifier(wav); 
echo_sig = amplifier(wav + jam_sig);  
jammer_sig = amplifier(jam_sig); 