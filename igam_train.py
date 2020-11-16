#IGAM train

from cifar_input import CIFAR100_Raw, CIFAR100_Augmented
from teacher_model_cifar import teacher
from pgd_attack import LinfPGDAttack

import numpy as np
import time
import os
import copy
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models, transforms
import torch.nn.functional as F

from torch.autograd import Variable

import config_train_igam


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def to_var(x, requires_grad=False, volatile=False):
    """
    Varialbe type that automatically choose cpu or cuda
    """
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=requires_grad, volatile=volatile)

#igamsource

class IGAM:

    def __init__(self, model_dir, data_path, target_class, lr, weight_decay, momentum, lr_boundaries, lr_decay, batch_size, 
                image_size, flip_rate, train_steps, beta, gamma, summary_steps):

        self.batch_size = batch_size
        self.train_steps = train_steps
        self.summary_steps = summary_steps
        self.beta = beta
        self.gamma = gamma

        #Load source model
        self.source_model = model_cifar(10)
        self.source_model.fc = nn.Linear(640, target_class)
        self.source_model = self.source_model.load_state_dict(torch.load(args.model_dir + 'finetune.pt'))
        self.source_model = self.source_model.to(device)

        #Student model
        self.std_model = model_cifar(target_class).to(device)

        #Discriminator
        self.disc_model = disc_model().to(device)

        self.std_optimizer = torch.optim.SGD(self.std_model.parameters(), lr, momentum, weight_decay)
        self.disc_optimizer = torch.optim.SGD(self.disc_model.parameters(), lr, momentum)

        self.std_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.std_optimizer, lr_boundaries, lr_decay)
        self.disc_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.disc_optimizer, lr_boundaries, lr_decay)

        self.loss_fn = torch.nn.CrossEntropyLoss()

        self.train = CIFAR100_Augmented(path, train=True, download=True, batch_size=self.batch_size, shuffle=True, 
                                            num_workers=2, pad=2, image_size=image_size, flip_rate=flip_rate)
        self.val = CIFAR100_Augmented(path, train=False, download=True, batch_size=self.batch_size, shuffle=True, 
                                            num_workers=2, pad=2, image_size=image_size, flip_rate=flip_rate)

        self.raw_train = CIFAR100_Raw(path, train=True, download=True, batch_size=self.batch_size, shuffle=True, 
                                            num_workers=2, pad=2, image_size=image_size, flip_rate=flip_rate)
        self.raw_val = CIFAR100_Raw(path, train=False, download=True, batch_size=self.batch_size, shuffle=True, 
                                            num_workers=2, pad=2, image_size=image_size, flip_rate=flip_rate)

    def train_model(self):

        since = time.time()

        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_acc = 0.0

        #source_input_grad = []
        #std_input_grad = []

        for i in range(self.train_steps):

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in self.train.trainloader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                source_inputs = to_var(inputs, requires_grad=True)
                std_inputs = to_var(inputs, requires_grad=True)

                self.optimizer.zero_grad()

                with torch.set_grad_enabled(True):

                    #source model
                    source_outputs = self.source_model(source_inputs) 

                    _, source_preds = torch.max(source_outputs, 1)

                    source_loss = self.loss_fn(source_outputs, labels)

                    # backward + optimize only if in training phase
                    source_loss.backward()
                    source_grad = source_inputs.grad.data.cpu().numpy()
                    #source_input_grad.append(ft_grad)
                    
                    #student model
                    std_outputs = self.std_model(std_inputs)

                    _, std_preds = torch.max(std_outputs, 1)

                    std_loss = self.loss_fn(std_outputs, labels)

                    # backward + optimize only if in training phase
                    std_loss.backward()
                    std_grad = std_inputs.grad.data.cpu().numpy()
                    #std_input_grad.append(std_grad)

                    #discriminator
                    disc_inputs = torch.cat([source_grad, std_grad], dim=0)
                    disc_labels = torch.tensor(np.concatenate((np.full(self.batch_size,1), np.full(self.batch_size,0))))

                    disc_outputs = self.disc_model(disc_inputs)

                    _, disc_preds = torch.max(disc_outputs, 1)

                    disc_loss = self.loss_fn(disc_outputs, disc_labels)

                    #optimize total loss    
                    input_grad_l2_norm_diff = ((F.normalize((source_grad - std_grad), p=2)) ** 2).mean()
                    total_loss = std_loss - self.beta * disc_loss + self.gamma * input_grad_l2_norm_diff

                    total_loss.backward()
                    self.std_optimizer.step()

                if i % disc_update_steps == 0:
                    disc_loss.backward()
                    self.disc_optimizer.step()

                std_running_loss += std_loss.item() * inputs.size(0)
                std_running_corrects += torch.sum(std_preds == labels.data)

                disc_running_loss += disc_loss.item() * disc_inputs.size(0)
                disc_running_corrects += torch.sum(disc_preds == disc_labels.data)

            self.std_scheduler.step()
            self.disc_scheduler.step()

            if i % self.summary_steps == 0:

                epoch_train_loss = std_running_loss / len(self.train.trainset)
                epoch_train_acc = std_running_corrects / len(self.train.trainset)

                disc_train_loss = disc_running_loss / len(self.train.trainset)
                disc_train_acc = disc_running_corrects / len(self.train.trainset)

                print('{} Train Loss: {:.4f} Trian Acc: {:.4f} Discriminator Loss: {:.4f} Discriminator Acc: {:.4f}' \
                    .format(i, epoch_train_loss, epoch_train_acc, disc_train_loss, disc_train_acc))

                running_loss = 0.0
                running_corrects = 0

                for val_inputs, val_labels in self.val.trainloader:
                    val_inputs = val_inputs.to(device)
                    val_labels = val_labels.to(device)

                    source_val_inputs = to_var(val_inputs, requires_grad=True)
                    std_val_inputs = to_var(val_inputs, requires_grad=True)

                    self.optimizer.zero_grad()

                    with torch.set_grad_enabled(False):

                        #source model
                        source_val_outputs = self.source_model(source_val_inputs) 

                        _, source_val_preds = torch.max(source_val_outputs, 1)

                        source_val_loss = self.loss_fn(source_val_outputs, val_labels)

                        # backward + optimize only if in training phase
                        source_val_loss.backward()
                        source_val_grad = source_val_inputs.grad.data.cpu().numpy()
                        
                        #student model
                        std_val_outputs = self.std_model(std_val_inputs)

                        _, std_val_preds = torch.max(std_val_outputs, 1)

                        std_val_loss = self.loss_fn(std_val_outputs, val_labels)

                        # backward + optimize only if in training phase
                        std_val_loss.backward()
                        std_val_grad = std_val_inputs.grad.data.cpu().numpy()
                        #std_input_grad.append(std_grad)

                        #discriminator
                        disc_val_inputs = torch.cat([source_val_grad, std_val_grad], dim=0)
                        disc_val_labels = torch.tensor(np.concatenate((np.full(self.batch_size,1), np.full(self.batch_size,0))))

                        disc_val_outputs = self.disc_model(disc_val_inputs)

                        _, disc_val_preds = torch.max(disc_val_outputs, 1)

                        disc_val_loss = self.loss_fn(disc_val_outputs, disc_val_labels)

                        #optimize total loss    
                        val_input_grad_l2_norm_diff = ((F.normalize((source_val_grad - std_val_grad), p=2)) ** 2).mean()
                        val_total_loss = std_val_loss - self.beta * disc_val_loss + self.gamma * val_input_grad_l2_norm_diff

                    val_std_running_loss += std_val_loss.item() * inputs.size(0)
                    val_std_running_corrects += torch.sum(std_val_preds == val_labels.data)

                    val_disc_running_loss += disc_val_loss.item() * disc_val_inputs.size(0)
                    val_disc_running_corrects += torch.sum(disc_val_preds == disc_val_labels.data)

                if epoch_train_acc > best_acc:
                    best_acc = epoch_train_acc
                    best_model_wts = copy.deepcopy(self.std_model.state_dict()) 

        torch.save(best_model_wts, arg.model_dir + "igam.pt")

            #print()

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc))

        # load best model weights
        self.std_model.load_state_dict(best_model_wts)

        # full test evaluation

        running_corrects = 0

        for raw_val_inputs, raw_val_labels in self.raw_val.trainloader:
            raw_val_inputs = raw_val_inputs.to(device)
            raw_val_labels = raw_val_labels.to(device)

            self.optimizer.zero_grad()

            with torch.set_grad_enabled(False):

                outputs = self.std_model(raw_val_inputs) 
                _, preds = torch.max(outputs, 1)

            running_corrects += torch.sum(preds == raw_val_labels.data)

        eval_acc = running_corrects / len(self.raw_val.trainset)

        print('Eval Acc: {:.4f}'.format(eval_acc))

        return self.std_model


if __name__ == '__main__':
    args = config_train_igam.get_args()
    args_dict = vars(args)
    train_model = IGAM(**args_dict)
    model = train_model.train_model()


