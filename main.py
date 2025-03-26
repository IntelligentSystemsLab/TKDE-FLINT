import torch
from server import Metanet
import time
import os
import numpy as np
import random
start = time.time()
from datetime import datetime

def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def main(method, flint, opentimes=None, seed=None, C=None, T=None):
    
    seed_torch(seed=seed)
    print(f'seed={seed},method={method},flint={flint},opentimes={opentimes}')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    meta_net = Metanet(device=device, dir=None)
    path = f'./results/'

    if method in ['FedAvg']:
        epoch = None
        meta_net.Testing(round=0)
        for i in range(epoch):
            print("{} round training.".format(i + 1))
            prev_time = datetime.now()
            meta_net.fedavg_train(round=i + 1)
            meta_net.Testing(round=i + 1)
            if opentimes == 'whole':
                k = epoch
            else:
                k = opentimes
            if flint == True and i < k:
                meta_net.sampling_proxy(round=i + 1, C=C)
                meta_net.reverse_distil(round=i + 1, T=T)
                meta_net.distil_meta(round=i + 1, T=T)
            cur_time = datetime.now()
            h, remainder = divmod((cur_time - prev_time).seconds, 3600)
            m, s = divmod(remainder, 60)
            time_str = "Time %02d:%02d:%02d" % (h, m, s)
            print(time_str)
            save_path = None
            meta_net.save_result_compact_model(save_path)

    elif method in ['FedProx']:
        epoch = None
        meta_net.Testing(round=0)
        for i in range(epoch):
            print("{} round training.".format(i + 1))
            prev_time = datetime.now()
            meta_net.fedprox_train(round=i + 1)
            meta_net.Testing(round=i + 1)
            if opentimes == 'whole':
                k = epoch
            else:
                k = opentimes
            if flint == True and i < k:
                meta_net.sampling_proxy(round=i + 1, C=C)
                meta_net.reverse_distil(round=i + 1, T=T)
                meta_net.distil_meta(round=i + 1, T=T)
            cur_time = datetime.now()
            h, remainder = divmod((cur_time - prev_time).seconds, 3600)
            m, s = divmod(remainder, 60)
            time_str = "Time %02d:%02d:%02d" % (h, m, s)
            print(time_str)
            save_path = None
            meta_net.save_result_compact_model(save_path)

    elif method in ['MOON']:
        epoch = None
        meta_net.Testing(round=0)
        for i in range(epoch):
            print("{} round training.".format(i + 1))
            prev_time = datetime.now()
            meta_net.moon_train(round=i + 1)
            meta_net.Testing(round=i + 1)
            if opentimes == 'whole':
                k = epoch
            else:
                k = opentimes
            if flint == True and i < k:
                meta_net.sampling_proxy(round=i + 1, C=C)
                meta_net.reverse_distil(round=i + 1, T=T)
                meta_net.distil_meta(round=i + 1, T=T)
            cur_time = datetime.now()
            h, remainder = divmod((cur_time - prev_time).seconds, 3600)
            m, s = divmod(remainder, 60)
            time_str = "Time %02d:%02d:%02d" % (h, m, s)
            print(time_str)
            save_path = None
            meta_net.save_result_compact_model(save_path)

'''
start my codes:
'''
if __name__ == '__main__':
    method_list = None
    for method in method_list:
        main(method=method, flint=False)
        main(method=method, flint=True)





