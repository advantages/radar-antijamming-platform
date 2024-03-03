function [reward,s_radar,s_echo,s_jammer]=overall_func(radar_act,jammer_act,ERP,Fc,urasize,IfBackBaffled,PRF,PW,SweepBW,NumPuls,NumSf,PeakPower,Gain,InUseOutputPort,MeanRCS,TwoWayPropagation_target,TwoWayPropagation_jammer,NoisePower,Target_loc,Jammer_loc,num_sp)



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
% URA.ElementSpacing
%% Pulse Patameters

NumPuls=double(NumPuls);
SweepBW=double(SweepBW);
PRF=double(PRF);


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
rng('default');  

waveform = phased.LinearFMWaveform('SampleRate',Fs,'PulseWidth',RarPara.PW,'PRF',RarPara.PRF,'SweepBandwidth',RarPara.SweepBW,'SweepDirection','Up',...
    'Envelope','Rectangular','OutputFormat','Pulses','NumPulses',RarPara.NumPuls,'SweepInterval','Positive');   

%% Initialize for each model
transmitter = phased.Transmitter('PeakPower',PeakPower,'Gain',Gain,'InUseOutputPort',InUseOutputPort); % Output power: 5e2*1e6=5e8 
radiator = phased.Radiator('Sensor',URA,'OperatingFrequency',Fc);  %  coherent sum of the delayed radiated fields from all elements  
jammer = barrageJammer('ERP',ERP,'SamplesPerFrame',waveform.NumPulses*waveform.SampleRate/waveform.PRF); % ERP: power % time for one pulse * sample rate 
target = phased.RadarTarget('Model','Nonfluctuating','MeanRCS',MeanRCS,'OperatingFrequency',Fc); % 10 m^2 
targetchannel = phased.FreeSpace('TwoWayPropagation',TwoWayPropagation_target,'SampleRate',Fs,'OperatingFrequency', Fc);    
jammerchannel = phased.FreeSpace('TwoWayPropagation',TwoWayPropagation_jammer,'SampleRate',Fs,'OperatingFrequency', Fc);   
collector = phased.Collector('Sensor',URA,'OperatingFrequency',Fc); 
% amplifier = phased.ReceiverPreamp('EnableInputPort',false); % all noise 
% amplifier = phased.ReceiverPreamp('EnableInputPort',true,NoiseFigure=1, SampleRate=Fs); % all noise  
amplifier = phased.ReceiverPreamp('NoiseMethod','Noise power','NoisePower',NoisePower); % all noise     
% Assume target and jammer are in the same place, but there is a time delay
% between radar and jammer 
targetloc = Target_loc;   
jammerloc = Jammer_loc;    
[~,tgtang] = rangeangle(targetloc); %  path length and direction angles  
[~,jamang] = rangeangle(jammerloc); 


% nsc = floor(Fs*RarPara.PW/32);
% nov = floor(nsc/2); 
% nff = max(1024,2^nextpow2(nsc));






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
% delay = zeros(floor(num_zeros/2),1); 
% jam_sig = [delay; jam_sig]; 
% jam_sig = jam_sig(1:end-floor(num_zeros/2)); % Already synchronize with radar
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
% figure;
% subplot(2,2,1);
% spectrogram(s_radar, hamming(nsc),nov,nff, Fs, 'yaxis');  
% ylim([0 10]);
% subplot(2,2,2);
% spectrogram(s_jammer, hamming(nsc),nov,nff, Fs, 'yaxis'); 
% ylim([0 10]);
% subplot(2,2,3); 
% spectrogram(s_echo, hamming(nsc),nov,nff, Fs, 'yaxis');   
% ylim([0 10]); 

% figure;
% % Plot the result, and compare it with received waveform with and without jamming.  
% subplot(2,1,1)
% t = unigrid(0,1/Fs,size(radar_sig,1)*1/Fs,'[)'); % time 
% plot(t*1000,real(radar_sig(:,1))) 
% title('Magnitudes of Pulse Waveform Without Jamming--Element 1')      
% ylabel('Magnitude')
% subplot(2,1,2)
% plot(t*1000,real(echo_sig(:,1))) 
% title('Magnitudes of Pulse Waveform with Jamming--Element 1')         
% xlabel('millisec')   
% ylabel('Magnitude')  

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
    
    
%     t_pulse = 1000*unigrid(0,1/Fs,length(echo_bp)*1/Fs,'[)'); % time
%     if ~isempty(fil_nojam)
%         figure;
%         plot(t_pulse,abs(fil_nojam{1})) 
%     end
% 
%     if ~isempty(fil_jam)
%         figure;
%         plot(t_pulse,abs(fil_jam{1})) 
%     end

% elseif flag_filter == 2
%     %% Matched Filtering-II
%     spec_echo = spectrogram(s_echo, hamming(nsc),nov,nff, Fs, 'yaxis');
%     spec_radar = spectrogram(radar_wave(1:len_pulse), hamming(nsc),nov,nff, Fs, 'yaxis');  
%     % coef_radar = fliplr(spec_radar);
%     coef_radar = flip(conj(spec_radar), 2);
%     % fil_output = conv2(spec_echo, coef_radar, 'same');
%     fil_output = zeros(size(spec_echo));
%     for i = 1:nff
%         fil_output(i,:) = conv2(spec_echo(i,:), coef_radar(i,:), 'same');
%     end

    %% Mask location
%     figure;
%     imagesc(abs(fil_output));
%     fil_smooth = imgaussfilt(abs(fil_output), 20);
%     figure;
%     imagesc(fil_smooth);
%     [Gmag, ~] = imgradient(fil_smooth, 'Sobel');
%     figure;
%     imagesc(Gmag);
%     largeValueIndices = find(Gmag > 1);
%     threshold = mean(Gmag(largeValueIndices));
%     binaryEdgeImage = Gmag > threshold;
%     figure;
%     imagesc(binaryEdgeImage); 
%     edges = edge(binaryEdgeImage, 'Canny');
%     figure;
%     imagesc(edges);  
%     % Find edge
%     rows = find(sum(edges, 2));
%     cols = find(sum(edges, 1));
%     cx = round((min(rows)+max(rows))/2);
%     cy = round((min(cols)+max(cols))/2);
%     left = find(edges(cx, 1:cy));
%     leftidx = (min(left)+max(left))/2;
%     right = find(edges(cx, cy+1:end))+cy;
%     rightidx = (min(right)+max(right))/2;
%     up = find(edges(1:cx, cy));
%     upidx = (min(up)+max(up))/2;
%     down = find(edges(cx+1:end, cy))+cx;
%     downidx = (min(down)+max(down))/2;
%     % create the mask
%     mask = ones(size(fil_output));
%     degree = 0.2;
%     upidx = round((1-degree)*upidx);
%     downidx = round((1+degree)*downidx);
%     leftidx = round((1-degree)*leftidx);
%     rightidx = round((1+degree)*rightidx);
%     mask(upidx:downidx, leftidx:rightidx) = 0;
%     figure;
%     imagesc(mask);  
%     
%     %% Matched filtering with Mask
%     spec_mask = spec_echo.*mask;
%     fil_mask = zeros(size(spec_mask));
%     for i = 1:nff
%         fil_mask(i,:) = conv2(spec_mask(i,:), coef_radar(i,:), 'same');
%     end
%     figure;
%     imagesc(abs(fil_mask));
%     fil_sum = sum(fil_mask(1:size(fil_output,1)/2,:), 1);
%     figure;
%     plot(abs(fil_sum));
end



end
