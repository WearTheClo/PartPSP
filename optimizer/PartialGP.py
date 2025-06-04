import random
import torch
import numpy as np

from torch import distributed
from torch.optim.optimizer import Optimizer,required


class extractorGP(Optimizer):
    def __init__(self, ext_params, ext_lr=required, args=None):
        if ext_lr is not required and ext_lr<0.0:
            raise ValueError("Invalid extractor learning rate")

        defaults=dict(lr = ext_lr, noise_lr = args.noi_lr)
        super().__init__(ext_params, defaults)

        self.B=args.B
        self.sync=args.sync
        self.sync_period=args.sync_period
        self.iter_cnt=0 
        self.rank=args.rank
        self.node_num=args.world_size
        self.communication_size=0

        self.dp_noise = args.dp_noise
        if self.dp_noise:
            self.budget = args.budget
            self.C = args.C
            self.lamb = args.lamb
        self.grad_clip = args.grad_clip
        self.clip_thre = args.clip_thre

        self.topo=args.topo
        self.weight=[0.0]*self.node_num
        self.WM=torch.Tensor([[0.0]*self.node_num]*self.node_num).cuda()
        self.aux_var=torch.Tensor([1.0]).cuda()
        self.aux_vec=torch.Tensor([0.0]*self.node_num).cuda()

        if self.rank>self.node_num:
            raise ValueError("Rank more than world size")

        print("extractorGP")


    def reset_communication_size(self):
        self.communication_size=0


    def add_communication_size(self,each_send_size):
        tem = 0
        for i in range(self.node_num):
            if i==self.rank:
                continue
            if self.WM[i][self.rank]!=0.0:
                tem = tem + 1
        self.communication_size += each_send_size * tem


    def get_acc_communication_size(self):
        return self.communication_size


    def __setstate__(self,state):
        super().__setstate__(state)


    def ave_weight(self):
        self.weight=[0.0]*self.node_num

        idx = self.iter_cnt % self.B
        for i in range(self.node_num):
            if self.topo[idx*self.node_num+i][self.rank]==0:
                continue
            self.weight[i]=1.0

        if self.sync and ((self.iter_cnt + 1)% self.sync_period == 0):
            self.weight=[1.0]*self.node_num

        out=sum(self.weight)
        for i in range(self.node_num):
            if i==self.rank or self.weight[i]==0.0:
                continue
            self.weight[i]=1.0/out
        self.weight[self.rank]=2.0-sum(self.weight)


    def grad_clip_func(self):
        grad_norm = torch.Tensor([0.0]).cuda()
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad_norm.add_(p.grad.data.abs().sum(), alpha=1.0)
        threshold =  float(grad_norm)/self.clip_thre
        if threshold > 1.0:
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is None:
                        continue
                    p.grad.data.div_(threshold)


    def _update_params(self):
        # gradient descent
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                param_state=self.state[p]
                d_p=p.grad.data
                if 'param_buff' not in param_state:
                    param_state['param_buff']=torch.clone(p).detach()

                param_state['param_buff'].add_(d_p,alpha=-group['lr'])


    def _consensus_params(self):
        for i in range(self.node_num):
            for j in range(self.node_num):
                if j==self.rank:
                    self.WM[i][j]=self.weight[i]
                else:
                    self.WM[i][j]=0.0
        distributed.barrier()
        distributed.all_reduce(self.WM)# 本轮通信矩阵

        for i in range(self.node_num):
            if i==self.rank:
                self.aux_vec[i]=float(self.aux_var)
            else:
                self.aux_vec[i]=0.0
        distributed.barrier()
        distributed.all_reduce(self.aux_vec)# 规约修正变量
        tem=torch.Tensor([0.0]).cuda()
        for i in range(self.node_num):
            tem.add_(self.aux_vec[i].mul(self.WM[self.rank][i]))
        self.aux_var=torch.clone(tem)# 修正变量的加权和

        with torch.no_grad():
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is None:
                        continue
                    param_state = self.state[p]
                    tem_param = torch.clone(param_state['param_buff']).detach().cuda()
                    tem_zero = torch.zeros_like(param_state['param_buff']).detach().cuda()
                    tem_param = torch.unsqueeze(tem_param, dim = 0)
                    tem_zero = torch.unsqueeze(tem_zero, dim = 0)
                    for i in range(self.node_num):
                        if i < self.rank:
                            tem_param = torch.cat((tem_zero, tem_param),dim = 0)
                        elif i > self.rank:
                            tem_param = torch.cat((tem_param, tem_zero),dim = 0)
                        else:
                            continue
                    distributed.barrier()
                    distributed.all_reduce(tem_param)

                    # tem_param 存储了当前层的全网参数，应该立刻共识
                    tem_state = torch.zeros_like(p.data).cuda()
                    for i in range(self.node_num):
                        tem_state.add_(tem_param[i].mul(self.WM[self.rank][i]))

                    param_state['param_buff']=torch.clone(tem_state)
                    p.data=param_state['param_buff'].div(self.aux_var)


    def sensitivity_counter(self):
        with torch.no_grad():
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is None:
                        continue
                    param_state=self.state[p]
                    param_state['param_buff'+str(self.rank)] = torch.clone(param_state['param_buff'])
                    for i in range(self.node_num):
                        if i == self.rank:
                            continue
                        param_state['param_buff'+str(i)] = torch.zeros_like(param_state['param_buff'])
                    
                    for i in range(self.node_num):
                        distributed.all_reduce(param_state['param_buff'+str(i)])

        L1norm_list = []
        for i in range(self.node_num):
            if i == self.rank:
                L1norm_list.append(0.0)
            else:
                L1norm = torch.Tensor([0.0]).cuda()
                with torch.no_grad():
                    for group in self.param_groups:
                        for p in group['params']:
                            if p.grad is None:
                                continue
                            param_state=self.state[p]
                            tem = param_state['param_buff'+str(self.rank)].add(param_state['param_buff'+str(i)],alpha=-1.0)
                            tem.abs_()
                            L1norm += tem.sum()
                L1norm_list.append(float(L1norm))
        node_sens = max(L1norm_list)
        net_sens = torch.Tensor([node_sens]).cuda()
        distributed.all_reduce(net_sens, distributed.ReduceOp.MAX)
        return float(net_sens)


    def sensitivity_estimator(self):
        # 需要超参C和lambda，在本地更新后，加噪前进行，顺序一定不能错
        if self.iter_cnt == 0 or (self.sync and (self.iter_cnt % self.sync_period == 0)): #初始化
            self.esti_sens = torch.Tensor([0.0]).cuda()
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is None:
                        continue
                    d_p=p.grad.data
                    self.esti_sens.add_(d_p.abs().sum(), alpha = 2*group['lr']*self.C) 
        else:
            self.esti_sens.mul_(self.lamb)
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is None:
                        continue
                    param_state=self.state[p]
                    d_p=p.grad.data
                    self.esti_sens.add_(d_p.abs().sum(), alpha = 2*group['lr']*self.C)
                    self.esti_sens.add_(param_state['last_noise'].abs().sum(), alpha = 2*self.lamb*group['noise_lr']*self.C)
        distributed.all_reduce(self.esti_sens, distributed.ReduceOp.MAX) #选取网络中最大的为评估敏感度
        return float(self.esti_sens)


    def add_noise(self, noise_scale):
        with torch.no_grad():
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is None:
                        continue
                    param_state=self.state[p]
                    if 'param_buff' not in param_state:
                        param_state['param_buff']=torch.clone(p).detach()

                    la_noise = torch.tensor(np.random.laplace(loc=0, scale=noise_scale, size=list(param_state['param_buff'].shape)),
                                            dtype=param_state['param_buff'].dtype).cuda()
                    param_state['param_buff'].add_(la_noise,alpha=group['noise_lr'])
                    param_state['last_noise']=torch.clone(la_noise)


    def local_step(self,closure=None):
        loss=None
        if closure is not None:
            loss=closure
        if self.grad_clip:
            self.grad_clip_func()
        self._update_params()
        return loss


    def global_step(self):
        if self.dp_noise:
            real_sens = self.sensitivity_counter()
            esti_sens = self.sensitivity_estimator()
            self.add_noise(esti_sens/self.budget)
        self.ave_weight()
        self._consensus_params()
        self.iter_cnt+=1
        return [real_sens, esti_sens]
