function jammer_wave = GenerateJam(jammer_act, radar_act, jam_sig, Fs, len_sp)
% Generate jamming signals according to jammer act
% flag = 1: spot jamming
% flag = 2: reveive 1 transmit 1
% flag = 3: receive 1 transmit 3
flag = jammer_act(1);
% Nonzero part
nonzero_1 = find(jam_sig ~= 0,1); 
% nonzero_2 = length(jam_sig) - find(flip(jam_sig)~=0, 1) + 1; 
jam_ori = jam_sig(nonzero_1:end); 

% Different jamming type
if flag == 1    
    fj = jammer_act(2);
    jam_modify = GetBandpass(jam_ori, fj, Fs);  

elseif flag == 2
    look_part = zeros(len_sp,1);
    jam_modify = jam_ori;
    jam_modify(1:len_sp) = look_part;
    jam_modify(2*len_sp+1:3*len_sp) = look_part;
    fj1 = radar_act(1);
    fj2 = radar_act(3);
    jam_modify(len_sp+1:2*len_sp) = GetBandpass(jam_ori(len_sp+1:2*len_sp), fj1, Fs);  
    jam_modify(3*len_sp+1:4*len_sp) = GetBandpass(jam_ori(3*len_sp+1:4*len_sp), fj2, Fs);  

elseif flag == 3
    look_part = zeros(len_sp,1);
    jam_modify = jam_ori;
    jam_modify(1:len_sp) = look_part;
    fj = radar_act(1);
    jam_modify(len_sp+1:4*len_sp) = GetBandpass(jam_ori(len_sp+1:4*len_sp), fj, Fs);

else
    jam_modify = jam_ori;
end

jammer_wave = jam_sig;
jammer_wave(nonzero_1:end) = jam_modify; 


    function new_wave=GetBandpass(wave,f, Fs)
        f_delta = 1.5e6; 
        bp_low = (f-1)*2.5e6+5e5;
        bp_high = bp_low+f_delta; 
        new_wave = bandpass(wave, [bp_low, bp_high], Fs); 
    end


end

