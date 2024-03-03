
function reward=gene_single_mat_val(radar_act,jammer_act,num_sf,num_sp)
pr = 1e9;
pj = 1e11;
% num_sf = 8;
% num_sp = 4;
pw = 60e-6;
bw = 20e6;
prf = 1e3;
num_pulse = 1;  


%% URA
Fc = 10e9; % Carrier Frequency 
element = phased.IsotropicAntennaElement('FrequencyRange', [0.75*Fc 1.25*Fc]); % creates an antenna element with an isotropic response pattern 
urasize = [10 20]; % Planner array 
taper = taylorwin(urasize(1))*taylorwin(urasize(2))'; % tapper weights
URA = phased.URA('Size',urasize, 'Element',element,'Taper',taper);  % URA 
URA.Element.BackBaffled = true; % no back 
lambda = physconst('lightspeed')/Fc; % wave length 
URA.ElementSpacing = [lambda/2 lambda/2]; % ElementSpacing 

%% Pulse Patameters
RarPara = struct();
RarPara.PRF = prf; % pulse repetion frequency 
RarPara.PW = pw; % pulse width
RarPara.SweepBW = bw; % sweep bandwidth width (total bandwidth for lfm) % Divided in three subpulses,each for 2MHz 
RarPara.NumPuls = num_pulse; % number of pulse    
RarPara.NumSf = num_sf;
RarPara.NumSp = num_sp;
Fs = 2*RarPara.SweepBW; %Sample Rate   
RarPara.Fs = Fs; 
rng('default');  

waveform = phased.LinearFMWaveform('SampleRate',Fs,'PulseWidth',RarPara.PW,'PRF',RarPara.PRF,'SweepBandwidth',RarPara.SweepBW,'SweepDirection','Up',...
    'Envelope','Rectangular','OutputFormat','Pulses','NumPulses',RarPara.NumPuls,'SweepInterval','Positive');   

%% Initialize for each model
transmitter = phased.Transmitter('PeakPower',pr,'Gain',60,'InUseOutputPort',true); % Output power: 5e2*1e6=5e8 
radiator = phased.Radiator('Sensor',URA,'OperatingFrequency',Fc);  %  coherent sum of the delayed radiated fields from all elements  
jammer = barrageJammer('ERP',pj,'SamplesPerFrame',waveform.NumPulses*waveform.SampleRate/waveform.PRF); % ERP: power % time for one pulse * sample rate 
target = phased.RadarTarget('Model','Nonfluctuating','MeanRCS',10,'OperatingFrequency',Fc); % 10 m^2 
targetchannel = phased.FreeSpace('TwoWayPropagation',true,'SampleRate',Fs,'OperatingFrequency', Fc);    
jammerchannel = phased.FreeSpace('TwoWayPropagation',false,'SampleRate',Fs,'OperatingFrequency', Fc);   
collector = phased.Collector('Sensor',URA,'OperatingFrequency',Fc); 
% amplifier = phased.ReceiverPreamp('EnableInputPort',false); % all noise 
% amplifier = phased.ReceiverPreamp('EnableInputPort',true,NoiseFigure=1, SampleRate=Fs); % all noise  
amplifier = phased.ReceiverPreamp('NoiseMethod','Noise power','NoisePower',1e-13); % all noise     
% Assume target and jammer are in the same place, but there is a time delay
% between radar and jammer 
targetloc = [10000 ; 5000; 0];   
jammerloc = [10000; 5000; 0];    
[~,tgtang] = rangeangle(targetloc); %  path length and direction angles  
[~,jamang] = rangeangle(jammerloc); 

%% Radar act to Transmit Signal 
nsc = floor(Fs*RarPara.PW/32); % size of each window   
nov = floor(nsc/2); 
nff = max(1024,2^nextpow2(nsc));  
wav_ori = waveform(); 
% figure;
% spectrogram(wav_ori, hamming(nsc),nov,nff, Fs, 'yaxis')  
% Generate Transmit Signal Referring to Radar Action 
[radar_wave, len_sp] = GenerateWave(radar_act, wav_ori, RarPara);  
% figure;
% spectrogram(radar_wave, hamming(nsc),nov,nff, Fs, 'yaxis') 
% Transmit waveform 
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
num_zeros = find(wav(:,1) ~= 0,1)-1; 
% nonzero_end = length(wav(:,1)) - find(flip(wav(:,1))~=0, 1) + 1;   
% figure;
% plot(abs(radar_wav)) 

%% Jamming Type and Jammer Signal 
% Generate the jamming signal 
jam_sig = jammer();
% % Synchronization with radar
num_stop = 1/2*length(jam_sig);
jam_sig(num_stop:end) = 0; 
delay = zeros(floor(num_zeros/2),1); 
jam_sig = [delay; jam_sig]; 
jam_sig = jam_sig(1:end-floor(num_zeros/2)); % Already synchronize with radar
jam_sig = GenerateJam(jammer_act, radar_act, jam_sig, Fs, len_sp);  
% num_nz = nnz(wav(:,1)); 
% jam_sig(num_nz+1:end) = 0; 
% num_stop = 1/2*length(jam_sig);
% jam_sig(num_stop:end) = 0; 
% jam_sig = GenerateJam_spot(jammer_act, jam_sig, Fs);  
% figure;
% spectrogram(jam_sig, hamming(nsc),nov,nff, Fs, 'yaxis')  
jam_sig = jammerchannel(jam_sig,jammerloc,[0;0;0],[0;0;0],[0;0;0]);   
jam_sig = collector(jam_sig,jamang); 

%% Received Mixed Signal
% radar_sig = amplifier(wav, ~txstatus); 
% echo_sig = amplifier(wav + jam_sig, ~txstatus);  
% jammer_sig = amplifier(jam_sig, ~txstatus); 
radar_sig = amplifier(wav); 
echo_sig = amplifier(wav + jam_sig);  
jammer_sig = amplifier(jam_sig); 


% figure; 
% periodogram(jam_sig(:,1), [], 1024, Fs, 'power', 'centered');   
% xlim([0 12]); 
NumPerPuls = Fs/RarPara.PRF; % Number of samples per pulse  
% NumNonzero = nonzero_end*2;

s_radar = radar_sig(1:NumPerPuls,1);  
s_jammer = jammer_sig(1:NumPerPuls,1);  
s_echo = echo_sig(1:NumPerPuls,1);


%% Matched Filtering-I
flag_filter = 1;
% para_sp1 = [2e5, 2.25e6]; 
% para_sp2 = [2.75e6, 4.75e6];   
% para_sp3 = [5.25e6, 7.25e6]; 
% para_sp4 = [7.75e6, 9.75e6]; 
% para_spall = {para_sp1, para_sp2, para_sp3, para_sp4}; 
len_pulse = 4*len_sp; 

if flag_filter == 1 
    [nums, idxs] = hist(radar_act, unique(radar_act));
    fil_nojam = {}; 
    fil_jam = {};
    reward = 0;
    % Whole pulse 
    for i=1:length(idxs)
        sf=idxs(i);
        % generate paras for bandpass
        f_delta = 1.5e6; 
        bp_low = (sf-1)*2.5e6+5e5;
        bp_high = bp_low+f_delta;
        para_sp = [bp_low, bp_high];
%         para_sp = para_spall{sf};
        echo_bp = bandpass(echo_sig(:,1), para_sp, Fs); 
        radar_bp = bandpass(radar_wave(1:len_pulse), para_sp, Fs);
        coef = flip(conj(radar_bp));
        Matchfilter = phased.MatchedFilter('Coefficients', coef);  
        fil_pulse = Matchfilter(echo_bp);
        max_val = max(abs(fil_pulse));
        if 1e-2*max_val > median(abs(fil_pulse))
            reward = reward + max_val;
            fil_nojam = [fil_nojam, fil_pulse];
        else
            fil_jam = [fil_jam, fil_pulse];
        end
    end
end


end

