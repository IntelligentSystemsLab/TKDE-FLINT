import torch
from torch import nn
from copy import deepcopy
from torch.utils.data import DataLoader
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F


class Client(nn.Module):
    def __init__(self,model,id,path,type_client,update_step,lr,device,mode,batch_size):
        super(Client, self).__init__()
        self.id = id
        self.update_step = update_step  ## task-level inner update steps
        self.net = deepcopy(model)
        self.local_lr = lr
        self.batch_size = batch_size
        self.loss_function = torch.nn.CrossEntropyLoss()
        self.mode = mode
        self.time = 0
        self.epoch = 0
        self.dataset = torch.load(path)
        self.datasize = len(self.dataset)
        # print(self.datasize)
        if mode == 'fed_train':
            self.support_loader = DataLoader(
                self.dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=0,
                drop_last=True
            )
        else:
            print("mode input error")
        self.size = int(len(self.dataset) * 1.0)
        self.optim = torch.optim.Adam(self.net.parameters(), lr=self.local_lr, weight_decay=1e-6)
        self.device = device
        self.softmax = nn.Softmax(dim=1)
        self.CE_loss = torch.nn.CrossEntropyLoss().cuda()
        self.label_cnt = np.zeros(shape=100)
        for item in range(len(self.dataset)):
            self.label_cnt[int(self.dataset[item][1])] += 1
        self.label_cnt = list(self.label_cnt)
        # self.weights = [max(self.label_cnt) / num for num in self.label_cnt]
        self.weights = []
        for num in self.label_cnt:
            if num > 0:
                self.weights.append(max(self.label_cnt) / num)
            else:
                self.weights.append(0)
        self.class_weights = torch.FloatTensor(self.weights).to(device)
        self.CE_loss_weights = torch.nn.CrossEntropyLoss(weight=self.class_weights).cuda()
        self.previous_net = model


    def forward(self):
        pass


    def fedavg_local_train(self):
        self.net.train()
        for _ in range(self.update_step):
            for im, label in self.support_loader:
                if torch.cuda.is_available():
                    im = im.to(self.device)
                    label = label.to(self.device)
                logits, _ = self.net(im)
                loss = self.CE_loss_weights(logits, label)
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

    def fedprox_local_train(self, global_net, mu):
        self.net.train()
        self.global_net = deepcopy(global_net)
        self.global_net.eval()
        global_weight_collector = list(self.global_net.cuda().parameters())
        for _ in range(self.update_step):
            for im, label in self.support_loader:
                if torch.cuda.is_available():
                    im = im.to(self.device)
                    label = label.to(self.device)
                logits, _ = self.net(im)
                loss = self.CE_loss_weights(logits, label)
                # for fedprox
                fed_prox_reg = 0.0
                for param_index, param in enumerate(self.net.parameters()):
                    fed_prox_reg += ((mu / 2) * torch.norm((param - global_weight_collector[param_index])) ** 2)
                loss += fed_prox_reg
                # backward
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

    def moon_local_train(self, global_net, mu, tau):
        self.global_net = deepcopy(global_net)
        cos = torch.nn.CosineSimilarity(dim=-1)
        self.net.train()
        self.global_net.train()
        self.previous_net.train()
        for _ in range(self.update_step):
            for x, target in self.support_loader:
                if torch.cuda.is_available():
                    x = x.to(self.device)
                    target = target.to(self.device)
                    x.requires_grad = False
                    target.requires_grad = False
                pro1, _ = self.net(x)
                loss1 = self.CE_loss_weights(pro1, target)
                pro2, _ = global_net(x)
                posi = cos(pro1, pro2)
                logits = posi.reshape(-1, 1)
                self.previous_net.cuda()
                pro3, _ = self.previous_net(x)
                nega = cos(pro1, pro3)
                logits = torch.cat((logits, nega.reshape(-1, 1)), dim=1)
                logits /= tau
                labels = torch.zeros(x.size(0)).cuda().long()
                loss2 = mu * self.CE_loss(logits, labels)
                loss = loss1 + mu * loss2
                # backward
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
        self.previous_net = deepcopy(self.net)


    def refresh(self, model):
        for w, w_t in zip(self.net.parameters(), model.parameters()):
            w.data.copy_(w_t.data)











      

        
    

        


    
    
 