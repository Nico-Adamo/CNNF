import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import os
import torch.optim as optim
import numpy as np
import math
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import shutil
from tensorboardX import SummaryWriter
from advertorch.attacks import GradientSignAttack, LinfPGDAttack, LinfSPSAAttack
from advertorch.context import ctx_noparamgrad_and_eval

class Evaluator:
    def __init__(self, device, model):
        self.device = device
        self.model = model
        self.tr_transforms = transforms.Compose([
                                    transforms.RandomCrop(28, padding=4),
									transforms.RandomHorizontalFlip(),
									transforms.Normalize([0.5] * 1, [0.5] * 1)])

    def clean_accuracy(self, clean_loader, test='last'):
        """ Evaluate the model on clean dataset. """
        self.model.eval()

        correct = 0
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(clean_loader):
                data, target = data.to(self.device), target.to(self.device)
                if (test=='last'):
                    output = self.model.run_cycles(data)
                elif(test=='average'):
                    output = self.model.run_average(data)
                else:
                    self.model.reset()
                    output = self.model(data)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        acc = correct / len(clean_loader.dataset)
        print('Clean Test Acc {:.3f}'.format(100. * acc))

        return acc

    def attack_pgd(self, clean_loader, epsilon=0.1, eps_iter=0.02, test='average', ete=False, nb_iter=7):
        """ Use PGD to attack the model. """

        self.model.eval()
        self.model.reset()

        if (ete==False):
            adv_func = self.model.forward_adv
        else:
            adv_func = self.model.run_cycles_adv

        adversary = LinfPGDAttack(
            adv_func, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=epsilon,
            nb_iter=nb_iter, eps_iter=eps_iter, rand_init=True, clip_min=-1.0, clip_max=1.0, targeted=False)

        correct = 0
        for batch_idx, (data, target) in enumerate(clean_loader):
            data, target = data.to(self.device), target.to(self.device)
            self.model.reset()
            with ctx_noparamgrad_and_eval(self.model):
                adv_images = adversary.perturb(data, target)

            if(test=='last'):
                output = self.model.run_cycles(adv_images)
            elif(test=='average'):
                output = self.model.run_average(adv_images)
            else:
                self.model.reset()
                output = self.model(adv_images)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

        acc = correct / len(clean_loader.dataset)
        print('PGD attack Acc {:.3f}'.format(100. * acc))

        return acc

    def attack_spsa(self, clean_loader, epsilon=0.1, test='average', ete=False, nb_iter=7):
        """ Use SPSA to attack the model. """

        self.model.eval()
        self.model.reset()
        if (ete==False):
            adv_func = self.model.forward_adv
        else:
            adv_func = self.model.run_cycles_adv

        adversary = LinfSPSAAttack(
            adv_func, loss_fn=nn.CrossEntropyLoss(reduction="none"), eps=epsilon,
            nb_iter=nb_iter, delta=0.01, nb_sample=128, max_batch_size=64, clip_min=-1.0, clip_max=1.0, targeted=False)

        correct = 0
        numofdata = 0
        for batch_idx, (data, target) in enumerate(clean_loader):
            # To speed up the evaluation of attack, evaluate the first 10 batches
            if(batch_idx < 10):
                data, target = data.to(self.device), target.to(self.device)
                self.model.reset()
                with ctx_noparamgrad_and_eval(self.model):
                    adv_images = adversary.perturb(data, target)

                if(test=='last'):
                    output = self.model.run_cycles(adv_images)
                elif(test=='average'):
                    output = self.model.run_average(adv_images)
                else:
                    self.model.reset()
                    output = self.model(adv_images)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                numofdata += data.shape[0]

        acc = correct / numofdata
        print('SPSA attack Acc {:.3f}'.format(100. * acc))

        return acc

    def attack_pgd_transfer(self, model_attacker, clean_loader, epsilon=0.1, eps_iter=0.02, test='average', nb_iter=7):
        """ Use adversarial samples generated against model_attacker to attack the current model. """

        self.model.eval()
        self.model.reset()
        model_attacker.eval()
        model_attacker.reset()
        adversary = LinfPGDAttack(
            model_attacker.forward_adv, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=epsilon,
            nb_iter=nb_iter, eps_iter=eps_iter, rand_init=True, clip_min=-1.0, clip_max=1.0, targeted=False)

        correct = 0
        for batch_idx, (data, target) in enumerate(clean_loader):
            data, target = data.to(self.device), target.to(self.device)
            self.model.reset()
            model_attacker.reset()
            with ctx_noparamgrad_and_eval(model_attacker):
                adv_images = adversary.perturb(data, target)

                if(test=='last'):
                    output = self.model.run_cycles(adv_images)
                elif(test=='average'):
                    output = self.model.run_average(adv_images)
                else:
                    self.model.reset()
                    output = self.model(adv_images)

            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

        acc = correct / len(clean_loader.dataset)
        print('PGD attack Acc {:.3f}'.format(100. * acc))

        return acc

    def ttt_accuracy(self, corrupted_loader, optimizer, per_image = False, batch_size = 64):
        """ Accuracy with test-time training. """

        self.model.train()
        self.model.reset()

        correct = 0
        numofdata = 0
        for batch_idx, (data, target) in enumerate(corrupted_loader):
            # To speed up the evaluation of attack, evaluate the first 20 batches
            if batch_idx < 10:
                data, target = data.to(self.device), target.to(self.device)
                self.model.reset()
                if per_image:
                    output_list = []
                    for im in data:
                        optimizer.zero_grad()
                        self.model.reset()
                        inputs = [self.tr_transforms(im) for _ in range(batch_size)]
                        inputs = torch.stack(inputs)
                        output, orig_feature, block1_all, block2_all, recon, recon_block1, recon_block2 = self.model.forward_backward_adv(inputs, inter=True)
                        output_list.append(torch.mean(output.unsqueeze(0), 1))
                        loss = (F.mse_loss(recon, orig_feature) + F.mse_loss(recon_block1, block1_all) + F.mse_loss(recon_block2, block2_all)) * 0.1 / 3 # Seperate blocks here?
                        loss.backward()
                        optimizer.step()
                    output = torch.cat(output_list, dim=0)
                else:
                    optimizer.zero_grad()
                    self.model.reset()
                    output, orig_feature, block1_all, block2_all, recon, recon_block1, recon_block2 = self.model.forward_backward_adv_cycles(data)
                    loss = (F.mse_loss(recon, orig_feature) + F.mse_loss(recon_block1, block1_all) + F.mse_loss(recon_block2, block2_all)) * 0.1 / 3 # Seperate blocks here?
                    loss.backward()
                    optimizer.step()

                pred = output.argmax(dim=1, keepdim=True)
                num_correct = pred.eq(target.view_as(pred)).sum().item()
                correct += num_correct
                #print("Batch {} - Num correct: {} - Total: {}".format(batch_idx, num_correct, correct))
                numofdata += data.shape[0]

        acc = correct / numofdata
        print('TTT Corrupted Acc {:.3f}'.format(100. * acc))

        return acc

    def corrupted_accuracy(self, corrupted_loader):
        """ Evaluate the model on corrupted dataset w/ one cycle. """
        self.model.eval()

        correct = 0
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(corrupted_loader):
                data, target = data.to(self.device), target.to(self.device)
                self.model.reset()
                output = self.model(data)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        acc = correct / len(corrupted_loader.dataset)
        print('Corrupted Test Acc {:.3f}'.format(100. * acc))

        return acc
