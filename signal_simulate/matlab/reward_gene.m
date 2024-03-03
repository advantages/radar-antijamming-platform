function [reward,fil_nojam,fil_jam]=reward_gene(radar_sig,jammer_sig,echo_sig,SweepBW,PRF,len_sp,radar_act,radar_wave)

Fs = 2*SweepBW;



NumPerPuls = Fs/PRF; % Number of samples per pulse  
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