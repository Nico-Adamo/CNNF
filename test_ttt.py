from __future__ import print_function
import argparse
import torch
import torch.optim as optim
from torchvision import datasets, transforms
from shutil import copyfile
from datetime import datetime
from cnnf.model_cifar import WideResNet
from cnnf.model_mnist import CNNF
from eval import Evaluator
from corrupt_transform import AddGaussianNoise
import numpy as np
import os

def main():
    parser = argparse.ArgumentParser(description='CNNF-TTT testing')
    parser.add_argument('--dataset', choices=['cifar10', 'fashion'],
                        default='cifar10', help='the dataset for training the model')
    parser.add_argument('--test', choices=['average', 'last'],
                        default='average', help='output averaged logits or logits from the last iteration')
    parser.add_argument('--csv-dir', default='results.csv',
                        help='Directory for Saving the Evaluation results')
    parser.add_argument('--model-dir', default='models',
                        help='Directory for Saved Models')

    args = parser.parse_args()
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    clean_dir = 'data/'

    # load in corrupted data
    if args.dataset=='cifar10':
        dataloader = torch.utils.data.DataLoader(
            datasets.CIFAR10(clean_dir, train=False, download=True, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])),
            batch_size=64, shuffle=True,
            num_workers=4, pin_memory=True)
        eps = 0.063
        eps_iter = 0.02
        nb_iter = 7

    elif args.dataset == 'fashion':
        dataloader = torch.utils.data.DataLoader(
            datasets.FashionMNIST(clean_dir, train=False, download=True, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
                AddGaussianNoise(0., 0.5),
            ])),
            batch_size=100, shuffle=True)
        eps = 0.025
        eps_iter = 0.071
        nb_iter = 7

    log_acc_path = args.csv_dir
    evalmethod = args.test
    model_dir = args.model_dir

    with open(log_acc_path, 'a') as f:
        f.write(',clean,pgd_first,pgd_last,spsa_first,spsa_last,transfer,')
        f.write('\n')

    # Model to evaluate
    if args.dataset=='cifar10':
        model_name = 'CNNF_2_cifar.pt'
        model = WideResNet(40, 10, 2, 0.0, ind=5, cycles=2, res_param=0.1).to(device)
    elif args.dataset == 'fashion':
        model_name = 'CNNF_1_fmnist.pt'
        model = CNNF(10, ind=2, cycles=1, res_param=0.1).to(device)

    model_path = os.path.join(model_dir, model_name)
    model.load_state_dict(torch.load(model_path))
    eval = Evaluator(device, model)
    corrupted_acc = eval.corrupted_accuracy(dataloader)

    optimizer = torch.optim.SGD(
          model.parameters(),
          0.05,
          momentum=0.9, weight_decay=5e-4)

    spsa_acc_ete = eval.ttt_accuracy(dataloader, optimizer, per_image = False, batch_size = 10)


    with open(log_acc_path, 'a') as f:
        f.write('%s,' % model_name)
        #f.write('%0.2f,' % (100. * clean_acc))
        #f.write('%0.2f,' % (100. * pgd_acc_first))
        # f.write('%0.2f,' % (100. * pgd_acc_ete))
        #f.write('%0.2f,' % (100. * spsa_acc_first))
        # f.write('%0.2f,' % (100. * spsa_acc_ete))
        # f.write('%0.2f,' % (100. * transfer_acc))
        f.write('\n')

if __name__ == '__main__':
    main()

