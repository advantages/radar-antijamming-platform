import torch
from collections import Counter
import numpy as np
from torch.nn import functional as F
import torch.optim.lr_scheduler as lr_scheduler
from .level_nn_model import GPT,GPTConfig
# import level_nn_model as model
# Remark
# 1) jammer's a_radar is still limited in this setting
# 2) still sub-pulse level
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
class DataGenerator:
    def __init__(self, num_sf, num_sp):
        self.num_sf = num_sf
        self.num_sp = num_sp
        self.det_sample = int(1e1)
        self.freq_sample = int(1e1)
        self.nofreq_sample = int(1e1)
        # self.det_list = np.arange(self.num_sf)
        self.freq2_list = [[0.9, 0.1], [0.8, 0.2], [0.7, 0.3], [0.6, 0.4], [0.5, 0.5]]
        self.freq3_list = [[0.8, 0.1, 0.1], [0.7, 0.2, 0.1], [0.6, 0.3, 0.1], [0.6, 0.2, 0.2], [0.5, 0.4, 0.1], [0.5, 0.3, 0.2], [0.4,0.4,0.2], [0.4, 0.3, 0.3]]
        self.type_list = ['det', 'freq1', 'freq2', 'freq3', 'nofreq']
        # self.type_list = ['nofreq']

    def GetData(self, length):
        X_train_all = []
        Y_train_all = []
        for type in self.type_list:
            X_train, Y_train = self.GetY(type, length)
            X_train_all.extend(X_train)
            Y_train_all.extend(Y_train)
        X_train_all = torch.tensor(X_train_all, dtype=torch.long)
        Y_train_all = torch.tensor(Y_train_all, dtype=torch.long)

        return X_train_all, Y_train_all


    def GetX(self, length, n_sample):
        X_train = [[np.random.randint(0, self.num_sf) for _ in range(length * self.num_sp)] for _ in range(n_sample)]

        return X_train

    def GetY(self, type, length):
        X_train = []
        Y_train = []
        if type == 'det':
            for i in range(length*self.num_sp):
                y_train = []
                x_train = self.GetX(length, self.det_sample)
                for j in range(len(x_train)):
                    y_train.append(self.jammer_det(x_train[j], i))
                X_train.extend(x_train)
                Y_train.extend(y_train)
        elif type == 'freq1':
            y_train = []
            x_train = self.GetX(length, self.freq_sample)
            for j in range(len(x_train)):
                y_train.append(self.jammer_freq1(x_train[j]))
            X_train.extend(x_train)
            Y_train.extend(y_train)
        elif type == 'freq2':
            for i in self.freq2_list:
                y_train = []
                x_train = self.GetX(length, self.freq_sample)
                for j in range(len(x_train)):
                    y_train.append(self.jammer_freq2(x_train[j], i))
                X_train.extend(x_train)
                Y_train.extend(y_train)
        elif type == 'freq3':
            for i in self.freq3_list:
                y_train = []
                x_train = self.GetX(length, self.freq_sample)
                for j in range(len(x_train)):
                    y_train.append(self.jammer_freq3(x_train[j], i))
                X_train.extend(x_train)
                Y_train.extend(y_train)
        elif type == 'nofreq':
            y_train = []
            x_train = self.GetX(length, self.nofreq_sample)
            for j in range(len(x_train)):
                y_train.append(self.jammer_nofreq(x_train[j]))
            X_train.extend(x_train)
            Y_train.extend(y_train)


        return X_train, Y_train


    def jammer_det(self, X, idx):  # History-dependent deterministic jammer
        X = np.array(X)
        pick_freq = X[idx]
        act_jammer = [pick_freq] * self.num_sp
        a_jammer = self.Act2Num_Jammer(act_jammer)

        return a_jammer

    def jammer_freq1(self, X): # History-dependent freq stationary jammer
        freq_static = self.Freqs_Static(X)  # Static on each frequency on this specific history
        sorted_freqs = sorted(freq_static, key=freq_static.get, reverse=True)
        largest_freq = sorted_freqs[:1]
        freq_prob = np.zeros(self.num_sf)
        freq_prob[largest_freq] = 1
        act = np.random.choice(np.arange(self.num_sf), 1, p=freq_prob)[0]  # focus on two most common frequencies
        act_jammer = [act] * self.num_sp
        a_jammer = self.Act2Num_Jammer(act_jammer)


        return a_jammer

    def jammer_freq2(self, X, idx): # History-dependent freq stationary jammer
        freq_static = self.Freqs_Static(X)  # Static on each frequency on this specific history
        sorted_freqs = sorted(freq_static, key=freq_static.get, reverse=True)
        largest_two_freqs = sorted_freqs[:2]
        freq_prob = np.zeros(self.num_sf)
        freq_prob[largest_two_freqs[0]] = idx[0]
        freq_prob[largest_two_freqs[1]] = idx[1]
        act = np.random.choice(np.arange(self.num_sf), 1, p=freq_prob)[0]  # focus on two most common frequencies
        act_jammer = [act] * self.num_sp
        a_jammer = self.Act2Num_Jammer(act_jammer)


        return a_jammer

    def jammer_freq3(self, X, idx): # History-dependent freq stationary jammer
        freq_static = self.Freqs_Static(X)  # Static on each frequency on this specific history
        sorted_freqs = sorted(freq_static, key=freq_static.get, reverse=True)
        largest_three_freqs = sorted_freqs[:3]
        freq_prob = np.zeros(self.num_sf)
        freq_prob[largest_three_freqs[0]] = idx[0]
        freq_prob[largest_three_freqs[1]] = idx[1]
        freq_prob[largest_three_freqs[2]] = idx[2]
        act = np.random.choice(np.arange(self.num_sf), 1, p=freq_prob)[0]  # focus on two most common frequencies
        act_jammer = [act] * self.num_sp
        a_jammer = self.Act2Num_Jammer(act_jammer)


        return a_jammer

    def jammer_nofreq(self, X): # History-dependent nofreq stationary jammer
        freq_static = self.Freqs_Static(X)  # Static on each frequency on this specific history
        static = list(freq_static.values())
        freq_prob = np.array(static) / np.array(static).sum()
        a_jammer = np.random.choice(np.arange(self.num_sf), 1, p=freq_prob)[0]

        return a_jammer

    def Freqs_Static(self, history):  # Static on each frequency on a_radar specific history
        freq_static = {i: 0 for i in range(self.num_sf)}
        collect = Counter(history)
        for f in collect.keys():
            freq_static[f] = collect[f]

        return freq_static

    def Act2Num_Jammer(self, act_j):

        return act_j[0]



class MLPP(nn.Module):

    def __init__(self, n_embd=3):
        super().__init__()
        self.c_fc    = nn.Linear(4*n_embd, 4*4 *n_embd)
        self.c_proj  = nn.Linear(4*4 * n_embd, n_embd)  #why four times?
        self.nonlin = nn.GELU()

    def forward(self, x):
        x=x.float()
        # print(x.shape)
        x = self.c_fc(x)
        x = self.nonlin(x)
        x = self.c_proj(x)
        return x

class MyDataset(Dataset):
    def __init__(self,num_sf,num_sp,max_his):
        self.num_sf=num_sf
        self.num_sp=num_sp
        self.max_his=max_his
        dataGene = DataGenerator(self.num_sf, self.num_sp)

        self.X_train_all, self.Y_train_all = dataGene.GetData(self.max_his)
        self.Y_train_all=self.Y_train_all.unsqueeze(-1)
    def __len__(self):
        return self.X_train_all.shape[0]

    def __getitem__(self, idx):
        X_train=self.X_train_all[idx,:]
        Y_train=self.Y_train_all[idx]
        return X_train,Y_train




class HisNN:
    def __init__(self, actions01, actions02, max_history, num_sf, num_sp, cuda_num):
        self.num_sf = num_sf
        self.num_sp = num_sp
        self.max_history = max_history
        self.actions_radar = actions01
        self.actions_jammer = actions02
        self.nb_actions_radar = len(actions01)
        self.nb_actions_jammer = len(actions02)
        self.epoch = 500
        self.num_update = 1000
        self.Historys_radar = [] # store all the history (in a_radar domain)
        self.Historys_jammer = []
        for i in range(max_history-1):
            self.Historys_radar.append([])
            self.Historys_jammer.append([])
        # Initialization for three transformers

        self.cuda_num=cuda_num
        self.iterCount=0

        self.gpt_list = self.Init_gpts()
        self.Pretrain()
        #self.filename='result.txt'
        #self.count=0

    def neural_network_predict(self, opp_action, select_len, recent_history, A_hat):
        # Generate Data from all the history
        self.Historys_radar[select_len - 1].append(recent_history[-1])
        self.Historys_jammer[select_len - 1].append(opp_action)
        self.iterCount=self.iterCount+1
        if len(recent_history) <= select_len:
            str_radar = np.ones(self.nb_actions_radar) / self.nb_actions_radar
        else:
            X_test = []
            # print(recent_history)
            recent_history = np.array(recent_history)
            his_test = recent_history[(-1) * select_len:]
            # his_test = recent_history[(-1) * 6:]
            for i in his_test:
                X_test.extend(self.Num2Act_Radar(i))
            X_test = torch.tensor(X_test, dtype=torch.long).unsqueeze(0).cuda(self.cuda_num)  # One new data point
            gpt = self.gpt_list[select_len-1][0].cuda(self.cuda_num)
            # print(X_test.shape)
            # print(X_test.shape)
            Y_pred = gpt(X_test)
            Y_pred=Y_pred.cpu()
            freq_prob = torch.softmax(Y_pred, dim=1)
            strategy_jammer = freq_prob.detach().numpy().reshape(-1, 1)
            # Transfer to strategy
            rew_hat = np.matmul(A_hat, strategy_jammer)
            a_radar = np.argmax(rew_hat)
            str_radar = np.zeros(self.nb_actions_radar)
            str_radar[a_radar] = 1

        # Update the neural network every num_update rounds
        #if (len(self.Historys_radar[select_len-1]) % self.num_update == 0) and (self.iterCount <=6000) :
        if len(self.Historys_radar[select_len-1]) % self.num_update == 0 :
            history_radar = self.Historys_radar[select_len-1]
            history_jammer = self.Historys_jammer[select_len-1]
            # if len(history_radar) <= self.num_dataset:
            #     pass
            # else:
            #     history_radar = history_radar[-1*self.num_dataset:]
            #     history_jammer = history_jammer[-1 * self.num_dataset:]
            X_train, Y_train = self.History2Data(history_radar, history_jammer, select_len)
            # Choose neural network
            #X_train=X_train[ -997: , :]
            #Y_train=Y_train[-997:]
            gpt, optimizer, scheduler = self.gpt_list[select_len-1]
            X_train=X_train.cuda(self.cuda_num)
            Y_train=Y_train.cuda(self.cuda_num)
            #print(X_train.shape)
            #print(Y_train.shape)
            for i in range(self.epoch):
                #Training
                #print(X_train.shape)
                #print(Y_train.shape)
                output = gpt(X_train)
                loss = F.cross_entropy(output, Y_train)

                #with open(self.filename,'a+') as f:
                #    f.write(str(self.count)+" ")
                #    f.write(str(loss.item())+" ")
                #    f.writelines("\n")
                #self.count+=1


                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
                #iters = (len(self.Historys_radar[select_len-1]) // self.num_update)*self.num_update+i
                #print('--Training--: history:', select_len, 'iter:', iters, 'loss:', loss.item())
            self.gpt_list[select_len-1] = (gpt, optimizer, scheduler)

        return str_radar


    def Init_gpts(self):
        config_1 = GPTConfig(1 * self.num_sp, self.num_sf, n_layer=6, n_head=4, n_embd=32, bias=False)
        gpt_1 = GPT(config_1).cuda(self.cuda_num)
        optimizer_1 = torch.optim.Adam(gpt_1.parameters(), lr=1e-3, betas=(0.9, 0.95), weight_decay=1e-3)
        scheduler_1 = lr_scheduler.CosineAnnealingLR(optimizer_1, T_max=self.epoch)
        # history length =2
        config_2 = GPTConfig(2 * self.num_sp, self.num_sf, n_layer=6, n_head=4, n_embd=32, bias=False)
        gpt_2 = GPT(config_2).cuda(self.cuda_num)
        optimizer_2 = torch.optim.Adam(gpt_2.parameters(), lr=1e-3, betas=(0.9, 0.95), weight_decay=1e-3)
        scheduler_2 = lr_scheduler.CosineAnnealingLR(optimizer_2, T_max=self.epoch)
        # history length =3
        config_3 = GPTConfig(3 * self.num_sp, self.num_sf, n_layer=6, n_head=4, n_embd=32, bias=False)
        gpt_3 = GPT(config_3).cuda(self.cuda_num)
        # gpt_3=MLPP(n_embd=4).cuda()
        # optimizer_3 = torch.optim.Adam(gpt_3.parameters(), lr=1e-3, betas=(0.9, 0.95),weight_decay=1e-3)
        # # optimizer_3 = torch.optim.SGD(gpt_3.parameters(), lr=1e-3, momentum=0.9, weight_decay=1e-1)
        # scheduler_3 = lr_scheduler.CosineAnnealingLR(optimizer_3, T_max=self.epoch)
        optimizer_3 = torch.optim.Adam(gpt_3.parameters(), lr=4e-4, betas=(0.9, 0.95), weight_decay=4e-4)
        scheduler_3 = lr_scheduler.CosineAnnealingLR(optimizer_3, T_max=self.epoch,eta_min=2e-4)



        config_4 = GPTConfig(4 * self.num_sp, self.num_sf, n_layer=6, n_head=4, n_embd=32, bias=False)
        gpt_4 = GPT(config_4).cuda(self.cuda_num)
        optimizer_4 = torch.optim.Adam(gpt_4.parameters(), lr=1e-3, betas=(0.9, 0.95), weight_decay=1e-3)
        scheduler_4 = lr_scheduler.CosineAnnealingLR(optimizer_4, T_max=self.epoch)

        config_5 = GPTConfig(5 * self.num_sp, self.num_sf, n_layer=6, n_head=4, n_embd=32, bias=False)
        gpt_5 = GPT(config_5).cuda(self.cuda_num)
        optimizer_5 = torch.optim.Adam(gpt_5.parameters(), lr=1e-3, betas=(0.9, 0.95), weight_decay=1e-3)
        scheduler_5 = lr_scheduler.CosineAnnealingLR(optimizer_5, T_max=self.epoch)

        config_6 = GPTConfig(6 * self.num_sp, self.num_sf, n_layer=6, n_head=4, n_embd=32, bias=False)
        gpt_6 = GPT(config_6).cuda(self.cuda_num)
        optimizer_6 = torch.optim.Adam(gpt_6.parameters(), lr=1e-3, betas=(0.9, 0.95), weight_decay=1e-3)
        scheduler_6 = lr_scheduler.CosineAnnealingLR(optimizer_6, T_max=self.epoch)


        gpt_list = [(gpt_1, optimizer_1, scheduler_1), (gpt_2, optimizer_2, scheduler_2),
                         (gpt_3, optimizer_3, scheduler_3),(gpt_4, optimizer_4, scheduler_4),(gpt_5, optimizer_5, scheduler_5),(gpt_6, optimizer_6, scheduler_6)]

        # gpt_list = [(gpt_1, optimizer_1, scheduler_1), (gpt_2, optimizer_2, scheduler_2),(gpt_3, optimizer_3, scheduler_3)]

        return gpt_list

    def History2Data(self, history_radar, history_jammer, select_len): # Pay attention to the matching pair
        len_history = len(history_radar)
        X_train, Y_train = [], []
        for i in range(0, len_history-select_len):
            X_train.append(self.Act2Data_Radar(history_radar[i:i+select_len]))
            Y_train.append(self.Num2Act_Jammer(history_jammer[i+select_len])[0])
        X_train = torch.tensor(X_train, dtype=torch.long)
        Y_train = torch.tensor(Y_train, dtype=torch.long)

        return X_train, Y_train

    def Act2Data_Radar(self, ar_set):
        x = []
        for i in ar_set:
            x.extend(self.Num2Act_Radar(i))

        return x

    def Num2Act_Radar(self, ar):
        x = []
        for _ in range(self.num_sp):
            x.append(ar % self.num_sf)  # Get the remainder of y when divided by 10
            ar = ar // self.num_sf  # Divide y by 10 to move to the next digit
        x.reverse()  # Reverse the order of elements in x
        return x

    def Num2Act_Jammer(self, aj):
        x = [aj]*self.num_sp

        return x

    def Pretrain(self):
        n_samples = 1000
        n_epoch = 5
        # For different gpt, Generate different data

        for i in range(2,self.max_history-1):
        # for i in range(2,2):
        #     print(i)
            self.dataset = MyDataset(self.num_sf, self.num_sp, i+1)
            self.train_dataloader = DataLoader(dataset=self.dataset, batch_size=1024, shuffle=True)
            gpt, optimizer, scheduler = self.gpt_list[i]
            # Generate data
            # Training

            # print(gpt)
            # Y_train = []
            # X_train = [[np.random.randint(0, self.num_sf) for _ in range((i+1)*self.num_sp)] for _ in range(n_samples)]
            # for j in range(n_samples):
            #     Y_train.append(self.jammer_nofreq(X_train[j]))
            # X_train = torch.tensor(X_train, dtype=torch.long).cuda()
            # Y_train = torch.tensor(Y_train, dtype=torch.long).cuda()
            # # Training
            # for j in range(n_epoch):
            #     output = gpt(X_train)
            #     loss = F.cross_entropy(output, Y_train)
            #     loss.backward()
            #     optimizer.step()
            #     optimizer.zero_grad()
            #     scheduler.step()
            # self.gpt_list[i] = (gpt, optimizer, scheduler)


            for j in range(n_epoch):
                for step, data in enumerate(self.train_dataloader):
                    X_train,Y_train=data
                    X_train=X_train.cuda(self.cuda_num)
                    Y_train=Y_train.cuda(self.cuda_num)
                    # print(X_train.shape,Y_train.shape,"record")
                    
                    output = gpt(X_train)
                    Y_train=Y_train.squeeze(-1)
                    loss = F.cross_entropy(output, Y_train)
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()
            self.gpt_list[i] = (gpt, optimizer, scheduler)


    def jammer_nofreq(self, data): # History-dependent nofreq stationary jammer
        freq_static = self.Freqs_Static(data)  # Static on each frequency on this specific history
        static = list(freq_static.values())
        freq_prob = np.array(static) / np.array(static).sum()
        a_jammer = np.random.choice(np.arange(self.num_sf), 1, p=freq_prob)[0]
        return a_jammer

    def Freqs_Static(self, history):  # Static on each frequency on a_radar specific history
        freq_static = {i: 0 for i in range(self.num_sf)}
        collect = Counter(history)
        for f in collect.keys():
            freq_static[f] = collect[f]
        return freq_static



