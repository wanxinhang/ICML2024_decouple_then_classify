import os
import warnings
import random
import torch
import configparser
from args import parameter_parser
import argparse
import random
import numpy as np
import torch
from train import train
if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    args = parameter_parser()
    if args.fix_seed:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
    datasets=['3sources']
    use_cuda = args.cuda and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    for i in  range(len(datasets)):
        args.dataset=datasets[i]
        train(args,device)