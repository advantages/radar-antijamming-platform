from collections import Counter
import numpy as np

# History dependent matrix
# For frequency-dependent matrix:
#   1) Matrix: Set \times Freqs
#   2) Set is generated after knowing the history length, num_sp and num_sf
#   3) Freqs: size equals to num_sf; sorted in the order of the number of apperance for each frequency
#   4) Jammer's action is first transformed into Freqs
#   5) Generated Jammer's action is just spot jamming

class Hisdep:
    def __init__(self, actions01, actions02, max_history, num_sf, num_sp):
        self.num_sf = num_sf
        self.num_sp = num_sp
        self.actions_radar = actions01
        self.actions_jammer = actions02
        self.nb_actions_radar = len(actions01)
        self.nb_actions_jammer = len(actions02)
        # Construct policy matrix without freq
        self.countMats_nofreq = []
        self.countMats_freq = []
        for i in range(max_history-2):  # largest: range(5)
            countmat_nofreq = np.zeros((np.power(self.nb_actions_radar, i), self.nb_actions_jammer), dtype=np.uint16)
            self.countMats_nofreq.append(countmat_nofreq)
        # When i=0, freq makes no sense
        for i in np.arange(1, max_history):
            countmat_freq = np.zeros((self.count_freqsets(i*self.num_sp, self.num_sf), self.nb_actions_jammer), dtype=np.uint16)
            self.countMats_freq.append(countmat_freq)

    def stationary(self, a_jammer, A_hat):
        # Update count_vec
        count_vec = self.countMats_nofreq[0][0]
        count_vec[a_jammer] += 1
        # Get strategy_jammer
        strategy_jammer = count_vec / np.sum(count_vec)
        # Make decision based on strategy_jammer
        rew_hat = np.matmul(A_hat, strategy_jammer)
        a_radar = np.argmax(rew_hat)
        str_radar = np.zeros(self.nb_actions_radar)
        str_radar[a_radar] = 1
        self.countMats_nofreq[0][0] = count_vec

        return str_radar

    # History dependent update without frequency
    def history_dependent_nofreq(self, a_jammer, select_len, recent_history, A_hat):
        # Update count_matrix
        recent_history = np.array(recent_history)
        if len(recent_history) <= select_len:
            str_radar = np.ones(self.nb_actions_radar)/self.nb_actions_radar
        else:
            previous_actions = recent_history[(-1) * (select_len + 1):-1]

            previous_num = self.Set2Num_nofreq(previous_actions)
            count_matrix = self.countMats_nofreq[select_len]

            count_matrix[previous_num, a_jammer] += 1
            # Get strategy_jammer
            present_actions = recent_history[(-1) * select_len:]
            present_num = self.Set2Num_nofreq(present_actions)
            count_vec = count_matrix[present_num, :]
            if np.sum(count_vec) == 0:  # if all zeros, choose uniform
                strategy_jammer = np.ones(self.nb_actions_jammer) / self.nb_actions_jammer
            else:
                strategy_jammer = count_vec / np.sum(count_vec)
            # Make decision based on strategy_jammer
            rew_hat = np.matmul(A_hat, strategy_jammer)
            a_radar = np.argmax(rew_hat)
            str_radar = np.zeros(self.nb_actions_radar)
            str_radar[a_radar] = 1
            self.countMats_nofreq[select_len] = count_matrix
            # if select_len == 2:
            #     print(strategy_jammer)

        return str_radar

    # History dependent update with frequency property
    def history_dependent_freq(self, a_jammer, select_len, recent_history, A_hat):
        # Get frequency actions
        recent_history = np.array(recent_history)
        if len(recent_history) <= select_len:
            str_radar = np.ones(self.nb_actions_radar)/self.nb_actions_radar
        else:
            previous_freqs = []
            for i in recent_history[(-1) * (select_len + 1):-1]:
                previous_freqs.extend(self.Num2Act_Radar(i))
            # previous_freqDict records the freq static on this specific history
            previous_num, previous_freqDict = self.Set2Num_freq(select_len, previous_freqs)
            count_matrix = self.countMats_freq[select_len - 1]
            # a_jammer -> act_jammer -> freq_jammer
            freq_jammer = self.Num2Act_Jammer(a_jammer)[0]
            idx_jammer = self.getidx_jammer(freq_jammer, previous_freqDict)
            count_matrix[previous_num, idx_jammer] += 1
            # Get strategy_jammer
            present_acts = []
            for i in recent_history[(-1) * select_len:]:
                present_acts.extend(self.Num2Act_Radar(i))
            present_num, present_freqDict = self.Set2Num_freq(select_len, present_acts)
            count_vec = count_matrix[present_num, :]
            if np.sum(count_vec) == 0:  # if all zeros, choose uniform
                freq_yhat = np.ones(self.num_sf) / self.num_sf
            else:
                freq_yhat = count_vec / np.sum(count_vec)
            strategy_jammer = self.get_yhat(freq_yhat, present_freqDict)
            # Make decision based on strategy_jammer
            rew_hat = np.matmul(A_hat, strategy_jammer)
            a_radar = np.argmax(rew_hat)
            str_radar = np.zeros(self.nb_actions_radar)
            str_radar[a_radar] = 1
            self.countMats_freq[select_len-1] = count_matrix

        return str_radar


    def Num2Act_Radar(self, ar):
        x = []
        for _ in range(self.num_sp):
            x.append(ar % self.num_sf)  # Get the remainder of y when divided by 10
            ar = ar // self.num_sf  # Divide y by 10 to move to the next digit
        x.reverse()  # Reverse the order of elements in x
        return x

    def Num2Act_Jammer(self, aj):
        x=[aj]*self.num_sp

        return x

    def Act2Num_Jammer(self, act):
        aj = act[0]

        return aj

    def Set2Num_nofreq(self, history_set):
        value = 0
        for i in range(len(history_set)):
            value += history_set[i] * np.power(self.nb_actions_radar, len(history_set)-i-1)

        return value

    # do not need to know the exact freq in the set
    def Set2Num_freq(self, select_len, previous_acts):
        # Create freq_id
        freq_set = []
        self.divide_freqs(select_len*self.num_sp, self.num_sf, [], freq_set)
        # Get freq_dict
        act_dict = Counter(previous_acts)
        freq_dict = {i: 0 for i in range(self.num_sf)}
        for i in act_dict.keys():
            freq_dict[i] = act_dict[i]
        freq_value = list(freq_dict.values())
        freq_value.sort(reverse=True)
        num = freq_set.index(freq_value)

        return num, freq_dict

    def getidx_jammer(self, freq_jammer, freq_dict):
        # sort the freq(key) of freq_dict by the number of apperance(value)
        freq_sort = [key for key, _ in sorted(freq_dict.items(), key=lambda x: x[1], reverse=True)]
        idx_jammer = freq_sort.index(freq_jammer)

        return idx_jammer

    def get_yhat(self, freq_yhat, freq_dict):
        y_hat = np.zeros(self.nb_actions_jammer)
        freq_sort = [key for key, _ in sorted(freq_dict.items(), key=lambda x: x[1], reverse=True)]
        for i in range(self.num_sf):
            act_jammer = [freq_sort[i]]*self.num_sp
            a_jammer = self.Act2Num_Jammer(act_jammer)
            y_hat[a_jammer] = freq_yhat[i]

        return y_hat


        # All possible sets for different history length
    def divide_freqs(self, M, N, current_combination, all_combinations):
        # M: number of subpulses(balls)
        # N: number of freqs(sets)
        if M == 0 and N == 0:
            # Base case: All balls have been divided into sets
            current_combination.sort(reverse=True)  # Sort the combination
            if current_combination not in all_combinations:
                all_combinations.append(current_combination)
        elif M > 0 and N > 0:
            # Recursive case: Divide a_radar ball into a_radar set and continue
            for i in range(M + 1):
                self.divide_freqs(M - i, N - 1, current_combination + [i], all_combinations)

    def count_freqsets(self, M, N):
        combinations = []
        self.divide_freqs(M, N, [], combinations)

        return len(combinations)





# if __name__ == "__main__":
#     his = Hisdep(range(27), range(7), 4, 10, 4)
#     # res = his.num_sorts
#     # res = his.Set2Num_Radar([26,26])
#     # combinations = []
#     # his.divide_freqs(9, 3,[],combinations)
#     # my_dict = dict(zip(range(len(combinations)), combinations))
#     # print(my_dict)
#     # res = his.Set2Num_freq(1,[1,3,3,6])
#     # print(res)
#     res = his.Num2Act_Radar(10)
#     print(res)