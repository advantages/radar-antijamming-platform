function [radar_wave, len_sp] = GenerateWave(radar_act, wave, RarPara)
% Generate new LFM according to radar's action
% Do not consider iterations first

%% Divided into Subpulses
num_sf = RarPara.NumSf;
num_sp = length(radar_act); 
n_sample = RarPara.Fs*RarPara.PW; % Fs*PW 
% n_sample
% fprintf("start\n")
% size(wave)
s_lfm = wave(1:n_sample);
% size(s_lfm)
% s_lfm = s_lfm/sqrt(sum(s_lfm.*s_lfm));
% Divide by num_sf
% mod(n_sample, num_sf)

% s_lfm(1:5)
% s_lfm(1:5*2)
% error("2222222")
% mod(n_sample, num_sf)=0



s_lfm = s_lfm(1:n_sample-mod(n_sample, num_sf)); 
n_lfm = length(s_lfm);
% n_lfm
len_sp = n_lfm/num_sf;

% len_sp

% size(s_lfm)

s_cut = reshape(s_lfm, len_sp, num_sf); 

% size(s_cut)

select_s = zeros(len_sp, num_sp); 

% size(select_s)


for n=1:num_sp
    select_s(:,n) = s_cut(:, radar_act(n)); 
end
radar_wave = zeros(size(wave));
radar_wave(1:len_sp*num_sp) = reshape(select_s,[],1); 
% size(reshape(select_s,[],1))
% len_sp*num_sp
% disp('Hello, World!');

end