import argparse
import math
import torch
import random
import numpy as np
import datetime
import torch.multiprocessing as mp

from fused_train import train_process
from topology.TopoLoador import TopoLoador

# mnist, fmnist, cifar10
# MLP,  resnet
# pretrain on torchvision

def main():
    parser = argparse.ArgumentParser(description='Decentralized image classification.')

    # basic config
    parser.add_argument('--model', type=str, default='MLP', help='model name, options: [MLP, Resnet18, Resnet34]')
    parser.add_argument('--shared_layers', type=int, default=1, help='number of shared layers, options: [1, 2, 3(full)]')
    parser.add_argument('--pretrained', action='store_false', help='load pretrained model (only for extractor). Default is True.')

    # data loader
    parser.add_argument('--data', type=str, default='mnist', help='dataset, options: [mnist, fmnist, cifar10]')
    parser.add_argument('--data_path', type=str, default='./data', help='root path of the data file')

    # evaluation
    parser.add_argument('--eva_store_path', type=str, default='./evaluation', help='evaluation store root path')

    # optimization
    parser.add_argument('--epochs', type=int, default=12, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=100, help='batch size of train input data')
    parser.add_argument('--ext_lr', type=float, default=0.1, help='optimizer learning rate for extractor')
    parser.add_argument('--cla_lr', type=float, default=0.1, help='optimizer learning rate for classifier')
    parser.add_argument('--noi_lr', type=float, default=0.001, help='noise rate')
    parser.add_argument('--ext_lr_decay', action='store_false', help='Set to enable learning rate decay for extractor. Default is True.')
    parser.add_argument('--cla_lr_decay', action='store_false', help='Set to enable learning rate decay for classifier. Default is True.')
    parser.add_argument('--noi_lr_decay', action='store_false', help='Set to enable noise rate decay. Default is True.')
    parser.add_argument('--ext_loc_ite', type=int, default=1, help='local optimization round for extractor')
    parser.add_argument('--cla_loc_ite', type=int, default=1, help='local optimization round for classifier')

    # dp noise
    parser.add_argument('--dp_noise', action='store_true', help='add noise in consensus. Default is False.')
    parser.add_argument('--budget', type=float, default=5.0, help='privacy budget hyperparameter')
    parser.add_argument('--C', type=float, default=0.5, help='hyperparameter C for sensitivity estimator')
    parser.add_argument('--lamb', type=float, default=0.3, help='hyperparameter lambda for sensitivity estimator')
    parser.add_argument('--grad_clip', action='store_true', help='gradient clipping trigger. Default is False.')
    parser.add_argument('--clip_thre', type=float, default=1.0, help='gradient clipping threshold')

    # decentralized device settings
    parser.add_argument(
        '-i',
        '--init-method',
        type=str,
        default='tcp://127.0.0.1:',
        help='URL specifying how to initialize the package.')
    parser.add_argument('--port', default=14562, type=int, help='port')
    parser.add_argument('-s', '--world-size', type=int, default=10, help='network Scale')
    parser.add_argument('--backend', type=str, default='gloo', help='Name of the backend to use.')

    # decentralized protocol settings
    parser.add_argument('--topology', type=str, default='./topology/10D5.txt')
    parser.add_argument('--strategy', type=str, default='Static') #Exponential Static
    parser.add_argument('--outdegree', default=2, type=int, help='outdegree for topology, only for Generate strategy')
    parser.add_argument('--B', default=1, type=int, help='strongly connected period for the protocol')
    parser.add_argument('--sync', action='store_false', help='force synchronization. Default is True.')
    parser.add_argument('--sync_period', default=4, type=int, help='force synchronization period to reset sensitivity')

    args = parser.parse_args()
    args.init_method += str(args.port)+"?use_libuv=False"

    args.eva_store_path += ('/' + args.data + '/fused/' + args.topology[11:15] + 'b' + str(int(args.budget)) + 'C' + str(args.C)
                            + 'L' + str(args.lamb)[2:] + 'SP' + str(args.sync_period) + args.model + 'SL' + str(args.shared_layers)
                            + 'elr' + str(args.ext_lr)[2:] + 'llr' + str(args.cla_lr)[2:] + 'nlr' + str(args.noi_lr)[2:]
                            )
    if args.grad_clip:
        args.eva_store_path += ('GCT' + str(int(args.clip_thre))) 

    # topo load
    if args.strategy == 'Exponential':
        args.B = int(math.log(args.world_size-1,2))+1
    args.topo = TopoLoador(args)

    print(args)

    mp.spawn(fn=train_process, args=(args,), nprocs=args.world_size)

    torch.cuda.empty_cache()

if __name__ == "__main__":
    fix_seed = 2024
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)
    main()
