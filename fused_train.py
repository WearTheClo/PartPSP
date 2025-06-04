# mnist, fmnist, cifar10
# MLP,  resnet
# pretrain on torchvision

import torch
import torch.nn as nn
import copy
import numpy as np

from torch import distributed
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets,transforms
from torch.utils import data

import models.bottleMLP as MLPloader
import models.torch_resnet_loader as tRESloader

import tools.AveModelTest as AMT
import evaluation.evaluate as eva
from optimizer.PartialGP import extractorGP
from optimizer.PEDFL import PEDFL


def freeze_params(model):
    for param in model.parameters():
        param.requires_grad = False


def unfreeze_params(model):
    for param in model.parameters():
        param.requires_grad = True


def train_process(rank,args):
    args.rank = rank
    ngpus_per_node = torch.cuda.device_count()
    gpu_id = args.rank % ngpus_per_node
    args.device=torch.device('cuda:'+str(gpu_id))
    torch.cuda.set_device(gpu_id)

    distributed.init_process_group(
        backend=args.backend,
        init_method=args.init_method,
        world_size=args.world_size,
        rank=args.rank,
    )

    num_classes = 10
    if args.data == 'mnist':
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        train_dataset = datasets.MNIST(root=args.data_path, train=True, download=False, transform=transform)
        test_dataset = datasets.MNIST(root=args.data_path, train=False, download=False, transform=transform)
    elif args.data == 'fmnist':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1))
        ])
        train_dataset = datasets.FashionMNIST(root=args.data_path, train=True, download=False, transform=transform)
        test_dataset = datasets.FashionMNIST(root=args.data_path, train=False, download=False, transform=transform)
    elif args.data == 'cifar10':
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        train_dataset = datasets.CIFAR10(root=args.data_path,train=True,download=False,transform=transform)
        test_dataset = datasets.CIFAR10(root=args.data_path,train=False,download=False,transform=transform)
    
    train_sampler = DistributedSampler(train_dataset, num_replicas=args.world_size, rank=args.rank, shuffle=True)

    train_loader=data.DataLoader(train_dataset,args.batch_size,sampler=train_sampler)
    test_loader=data.DataLoader(test_dataset,args.batch_size)
    
    if args.model == 'MLP':
        ext_model, cla_model = MLPloader.fused_torch_bottmlp_loader(in_features=784, num_classes=num_classes, shared = args.shared_layers, pretrained=args.pretrained) #mnist专用
    elif args.model == 'Resnet18':
        ext_model, cla_model = tRESloader.fused_torch_resnet18_loader(num_classes=num_classes, shared = args.shared_layers, pretrained=args.pretrained)
    elif args.model == 'Resnet34':
        ext_model, cla_model = tRESloader.fused_torch_resnet34_loader(num_classes=num_classes, shared = args.shared_layers, pretrained=args.pretrained)

    ext_model.to(args.device)
    cla_model.to(args.device)

    loss_fun=nn.CrossEntropyLoss()

    ext_optimizer = extractorGP(ext_model.parameters(), ext_lr=args.ext_lr, args=args)
    cla_optimizer = torch.optim.SGD(cla_model.parameters(), lr=args.cla_lr)
    
    e = eva.evaluate(args)
    L1_list = []
    
    for i in range(args.epochs):
        print("epoch number: {}".format(i+1))
        train_acc = 0.0
        train_loss = 0.0
        train_size = 0
        test_acc = 0.0
        test_loss = 0.0
        test_size = 0
        
        if (i+1)%5==0 and args.noi_lr_decay:
            args.noi_lr /= 10
            print("noise step size decay to {}".format(args.noi_lr))
        if (i+1)%5==0 and args.ext_lr_decay:
            args.ext_lr /= 10
            ext_optimizer = extractorGP(ext_model.parameters(), ext_lr=args.ext_lr, args=args)
            print("extractor's step size decay to {}".format(args.ext_lr))
        if (i+1)%5==0 and args.cla_lr_decay:
            args.cla_lr /= 10
            cla_optimizer = torch.optim.SGD(cla_model.parameters(), lr=args.cla_lr)
            print("classifier's step size decay to {}".format(args.cla_lr))
        
        ave_model = copy.deepcopy(ext_model)
        ave_optimizer = AMT.AVEModelTest(params=ave_model.parameters(), args=args)
        ave_optimizer.ave_params()

        ave_model.eval()
        cla_model.eval()
        for b,(x_test,y_test) in enumerate(test_loader):
            x_test=x_test.to(args.device)
            y_test=y_test.to(args.device)
            y_ave_med=ave_model(x_test)
            y_val=cla_model(y_ave_med)
            predicted=torch.max(y_val.data,1)[1]
            loss = loss_fun(y_val,y_test)

            test_loss += loss.item()
            test_acc += (predicted == y_test).sum()
            test_size += y_test.shape[0]

        e.append(loss_item=test_loss/test_size, c_r=100*test_acc/test_size, epoch=i, L2Dis=0)
        e.all_reduce()
        e.read_save()
        print('Rank:%d Test accuracy%10.7f%%'%(args.rank,100*test_acc/test_size))

        ext_model.train()
        cla_model.train()
        for b,(x_train,y_train) in enumerate(train_loader):
            b+=1
            x_train=x_train.to(args.device)
            y_train=y_train.to(args.device)

            # local cla update
            freeze_params(ext_model)
            for cla_loc in range(args.cla_loc_ite):
                y_ext = ext_model(x_train)
                y_cla = cla_model(y_ext)
                loss = loss_fun(y_cla,y_train)
                cla_optimizer.zero_grad()
                loss.backward()
                cla_optimizer.step() # only optimize cla
            unfreeze_params(ext_model)

            # local ext update
            for ext_loc in range(args.ext_loc_ite):
                y_ext = ext_model(x_train)
                y_cla = cla_model(y_ext)
                loss = loss_fun(y_cla,y_train)
                ext_optimizer.zero_grad()
                cla_optimizer.zero_grad()
                loss.backward()
                ext_optimizer.local_step() # only optimize ext

            # ext consensus, keep gradient
            real_sens, esti_sens = ext_optimizer.global_step()

            # train test
            y_ext = ext_model(x_train)
            y_pred = cla_model(y_ext)
            predicted=torch.max(y_pred.data,1)[1]
            loss = loss_fun(y_pred,y_train)

            train_loss += loss.item()
            train_acc += (predicted == y_train).sum()
            train_size += y_train.shape[0]
            # training L1
            L1_list.append([real_sens, esti_sens])

            if b%50==0:
                train_acc_sum = train_acc/train_size
                distributed.all_reduce(train_acc_sum)
                train_acc_sum.div_(args.world_size)
                train_loss_sum = train_loss/train_size
                if rank==0:
                    print(f'epoch:{i:2} batch:{b:2} Train loss:{train_loss_sum:10.8f} Train accuracy:{100*train_acc_sum:10.8f}')

    if rank==0:
        L1_array = np.array(L1_list)
        np.save(args.eva_store_path + '.npy', L1_array)
        print(np.mean(L1_array[:, 0]), np.mean(L1_array[:, 1]))
        real_bigger_esti = 0
        for row in L1_array:
            real, esti = row
            if real > esti:
                real_bigger_esti += 1
        print(real_bigger_esti)
