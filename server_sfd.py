import torch
from torchvision import models
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import pandas as pd
import os
from client_sfd import Client
from torch.utils.data import random_split
from copy import deepcopy
import random
from datetime import datetime
import torch.nn.functional as F
from torch.utils.data import DataLoader

def get_model_1():
  v = models.vgg19(pretrained = True)
  return v

class Adapter_1(nn.Module):
  def __init__(self, model_bone):
    super(Adapter_1, self).__init__()
    self.bone = model_bone
    self.linear = nn.Linear(1000, 10)

  def forward(self, input):
    hidden = self.bone(input)
    logits = self.linear(hidden)
    return logits, hidden

def get_model_2():
  v = models.mobilenet_v2(pretrained = True) # or False
  return v


class Adapter_2(nn.Module):
  def __init__(self, model_bone):
    super(Adapter_2, self).__init__()
    self.bone = model_bone
    self.linear = nn.Linear(1000, 10)

  def forward(self, input):
    hidden = self.bone(input)
    logits = self.linear(hidden)
    return logits, hidden


class Metanet(nn.Module):
  def __init__(self, device, dir, local_train_epoch=None, lr=5e-4):
    super(Metanet, self).__init__()
    self.device = device
    self.local_metatrain_epoch = local_train_epoch
    self.net_1_bone = get_model_1() #大模型骨干
    self.adapter_1 = Adapter_1
    self.net_1 = self.adapter_1(self.net_1_bone)  # 大模型
    self.net_2_bone = get_model_2()  # 小模型骨干
    self.adapter_2 = Adapter_2
    self.net_2 = self.adapter_2(self.net_2_bone)  # 小模型
    self.net_1_bone.to(self.device)
    self.net_2_bone.to(self.device)
    self.net_1.to(self.device)
    self.net_2.to(self.device)
    self.loss_function = torch.nn.CrossEntropyLoss(reduction='mean')
    self.clients = []
    self.test_clients = []
    self.mode_1 = "fed_train"
    self.mode_2 = "fed_test"
    self.batch_size = 64
    self.path_now = os.path.dirname(__file__)
    self.last_path = '/results_data'
    self.loss_list = []
    self.acc_list = []
    self.recall_list = []
    self.pre_list = []
    self.f1_list = []
    self.time_accum = [0]
    self.ave_loss_list = []
    self.ave_acc_list = []
    self.server_model_loss_list = []
    self.server_model_acc_list = []
    self.softmax = nn.Softmax(dim=1)
    self.KLloss = nn.KLDivLoss(reduction='batchmean').cuda()
    self.MSEloss = nn.MSELoss(reduction='mean').cuda()
    self.CElossSUM = nn.CrossEntropyLoss(reduction='sum')
    self.server_model_train_optim = torch.optim.Adam(self.net_1.parameters(), lr=1e-6, weight_decay=1e-6)
    self.distil_optim = torch.optim.Adam(self.net_2.parameters(), lr=lr, weight_decay=1e-6)
    self.lr = lr
    self.sd = None
    self.ht = None
    self.active = None

    '''
    Federated Learning
    '''
    self.proxy_data_all = torch.load(r'./proxy_dataset_10k.pt')

    '''
    alpha dir
    '''
    alpha = None
    train_path = r"./sfd_data/train"
    train_file_set = os.listdir(train_path)
    train_path_set = [os.path.join(train_path, i) for i in train_file_set]

    for index, path in enumerate(train_path_set):
      net = self.net_2.to(self.device)
      self.clients.append(
        Client(net, index, path, 'train', local_train_epoch, lr, self.device, self.batch_size))

    test_path = r"./sfd_data/test"
    test_file_set = os.listdir(test_path)
    test_path_set = [os.path.join(test_path, i) for i in test_file_set]

    for index, path in enumerate(test_path_set):
        net = None
        local_train_epoch = None
        self.test_clients.append(
            Client(net, index, path, 'test', local_train_epoch, lr, self.device, self.batch_size))


  def forward(self):
    pass

  def save_time(self,save_path):
    dataframe = pd.DataFrame(list(self.time_accum), columns=['time_accum'])
    dataframe.to_excel(save_path, index=False)


  def fedavg_train(self,round):

    id_train_0 = list(range(len(self.clients)))
    self.id_train = random.sample(id_train_0, int(len(id_train_0)*self.active)) #clients of this round
    for id,j in enumerate(self.id_train):
        self.clients[j].refresh(self.net_2)
        self.clients[j].fedavg_local_train()
        self.clients[j].epoch = round

    weight = []
    for id,j in enumerate(self.id_train):
      weight.append(self.clients[j].size)
    weight = np.array(weight)
    weight = weight / weight.sum()
    for id,j in enumerate(self.id_train):
      for w,w_t in zip(self.net_2.parameters(),self.clients[j].net.parameters()):
        if (w is None or id == 0):
          w_tem = Variable(torch.zeros_like(w)).to(self.device)
          w.data.copy_(w_tem.data)
        if w_t is None:
          w_t = Variable(torch.zeros_like(w)).to(self.device)
        w.data.add_(w_t.data*weight[id])


  def fedprox_train(self,round):

    id_train_0 = list(range(len(self.clients)))
    self.id_train = random.sample(id_train_0, int(len(id_train_0)*self.active)) #clients of this round
    for id,j in enumerate(self.id_train):
        self.clients[j].refresh(self.net_2)
        self.clients[j].fedprox_local_train(self.net_2, mu=0.001)
        self.clients[j].epoch = round

    weight = []
    for id,j in enumerate(self.id_train):
      weight.append(self.clients[j].size)
    weight = np.array(weight)
    weight = weight / weight.sum()
    for id,j in enumerate(self.id_train):
      for w,w_t in zip(self.net_2.parameters(),self.clients[j].net.parameters()):
        if (w is None or id == 0):
          w_tem = Variable(torch.zeros_like(w)).to(self.device)
          w.data.copy_(w_tem.data)
        if w_t is None:
          w_t = Variable(torch.zeros_like(w)).to(self.device)
        w.data.add_(w_t.data*weight[id])


  def moon_train(self, round):

      id_train_0 = list(range(len(self.clients)))
      self.id_train = random.sample(id_train_0, int(len(id_train_0) * self.active))  # clients of this round
      for id, j in enumerate(self.id_train):
          self.clients[j].refresh(self.net_2)
          self.clients[j].moon_local_train(self.net_2, mu=0.1, tau=0.5)
          self.clients[j].epoch = round

      weight = []
      for id, j in enumerate(self.id_train):
          weight.append(self.clients[j].size)
      weight = np.array(weight)
      weight = weight / weight.sum()
      for id, j in enumerate(self.id_train):
          for w, w_t in zip(self.net_2.parameters(), self.clients[j].net.parameters()):
              if (w is None or id == 0):
                  w_tem = Variable(torch.zeros_like(w)).to(self.device)
                  w.data.copy_(w_tem.data)
              if w_t is None:
                  w_t = Variable(torch.zeros_like(w)).to(self.device)
              w.data.add_(w_t.data * weight[id])


  def reverse_distil(self,round,T):

    device = self.device
    self.net_1.train()
    self.net_2.eval()
    for param in self.net_1_bone.parameters():
      param.requires_grad = False  
    prev_time = datetime.now()
    self.net_tem = deepcopy(self.net_1)
    net_tem = self.net_tem.to(device)
    net_tem.eval()

    for im in self.proxy_loader:
        im = im.to(device)  # (bs, 3, h, w)
        logits_1, hidden_1 = self.net_1(im)
        logits_prev, hidden_prev = net_tem(im)
        logits_2, hidden_2 = self.net_2(im)
        main_loss = self.KLloss(F.log_softmax(logits_1 / T), self.softmax(logits_2 / T))
        self_loss = self.KLloss(F.log_softmax(logits_1 / T), self.softmax(logits_prev / T))
        loss = main_loss + self.sd * self_loss
        # backward
        self.server_model_train_optim.zero_grad()
        loss.backward()
        self.server_model_train_optim.step()
    cur_time = datetime.now()
    h, remainder = divmod((cur_time - prev_time).seconds, 3600)
    m, s = divmod(remainder, 60)
    time_str = "Time %02d:%02d:%02d" % (h, m, s)



  def distil_meta(self,round,T):
    ########蒸馏##########
    self.net_1.eval()
    self.net_2.train()
    device = self.device
    prev_time = datetime.now()
    net_tem = deepcopy(self.net_2)
    net_tem.to(device)
    net_tem.train()
    meta_optim_inner = torch.optim.Adam(net_tem.parameters(), lr=self.lr, weight_decay=1e-6)

    for x in self.proxy_loader:
        if torch.cuda.is_available():
            x = x.to(device)
        logits_2, hidden_2 = net_tem(x)
        logits_1, hidden_1 = self.net_1(x)
        softloss = self.KLloss(F.log_softmax(logits_2 / T), self.softmax(logits_1 / T))
        hiddenloss = self.MSEloss(hidden_2, hidden_1)
        loss = softloss + self.ht * hiddenloss
        meta_optim_inner.zero_grad()
        loss.backward()
        meta_optim_inner.step()

    # backward
    self.distil_optim.zero_grad()
    for w, w_t in zip(self.net_2.parameters(), net_tem.parameters()):
        if w.grad is None:
            w.grad = Variable(torch.zeros_like(w)).to(device)
        w.grad.data.add_(w.data - w_t.data)
    self.distil_optim.step()

    cur_time = datetime.now()
    h, remainder = divmod((cur_time - prev_time).seconds, 3600)
    m, s = divmod(remainder, 60)
    time_str = "Time %02d:%02d:%02d" % (h, m, s)



  def sampling_proxy(self, round, C):
      length = int(C * len(self.proxy_data_all))
      proxy_data, _ = torch.utils.data.random_split(self.proxy_data_all,
                                                    [length, len(self.proxy_data_all)-length])
      self.proxy_loader = torch.utils.data.DataLoader(proxy_data,
                                                 batch_size=64,
                                                 shuffle=True,
                                                 num_workers=0,
                                                 drop_last=True)

  def save_result_compact_model(self, save_path):
      result = list(zip(self.ave_loss_list,
                        self.ave_acc_list))
      dataframe = pd.DataFrame(result, columns=['compact_model_loss',
                                                'compact_model_acc'])
      dataframe.to_excel(save_path, index=False)

  def Testing(self, round):
        net = self.net_2.to(self.device)
        id_test_0 = list(range(len(self.test_clients)))
        self.id_test = random.sample(id_test_0, int(len(id_test_0) * 1))  # clients of this round
        loss = 0.0
        acc = 0.0
        for id, j in enumerate(self.id_test):
            add_loss, add_acc = self.test_clients[j].client_test(net)
            loss += add_loss
            acc += add_acc
        ave_loss = loss/len(self.test_clients)
        ave_acc = acc/len(self.test_clients)
        print(f'compact model epoch_{round}, loss:{ave_loss}, acc:{ave_acc}')
        self.ave_loss_list.append(ave_loss)
        self.ave_acc_list.append(ave_acc)






















