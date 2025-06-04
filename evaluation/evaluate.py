import torch
from torch import distributed

import os
import time
import numpy as np

class evaluate():
    def __init__(self,args):
        self.loss_item=[]
        self.c_r=[]
        self.epoch=[]
        self.L2Dis=[]
        self.args=args

    def append(self,loss_item,c_r,epoch,L2Dis):
        self.loss_item.append(loss_item)
        self.c_r.append(c_r)
        self.epoch.append(epoch)
        self.L2Dis.append(L2Dis)

    def all_reduce(self):
        value=torch.Tensor([self.loss_item[-1]/self.args.world_size,self.c_r[-1]/self.args.world_size,self.L2Dis[-1]/self.args.world_size]).cuda()
        distributed.all_reduce(value)
        value=value.cpu().float()
        self.loss_item[-1],self.c_r[-1],self.L2Dis[-1]=value[0],value[1],value[2]

    def read_save(self):
        if self.args.rank!=0:
            return

        path_wor=self.args.eva_store_path+'.txt'
        
        with open(path_wor,'a') as f:
            f.write('epoch:%d Loss:%10.7f Accuracy:%10.7f%% L2Dis:%10.7f\n'%(self.epoch[-1],self.loss_item[-1],self.c_r[-1],self.L2Dis[-1]))
            f.close()

    def np_save(self):
        if self.args.rank!=0:
            return

        path_np=self.args.eva_store_path

        e=np.array(self.epoch)
        l=np.array(self.loss_item)
        c=np.array(self.c_r)
        d=np.array(self.L2Dis)

        np.save(path_np+'Epoch.npy',e)
        np.save(path_np+'Loss.npy',l)
        np.save(path_np+'CR.npy',c)
        np.save(path_np+'Distance.npy',d)
