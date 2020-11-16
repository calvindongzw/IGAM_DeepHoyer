#Teacher Model Finetune

from cifar_input import CIFAR10_Raw, CIFAR10_Augmented
from teacher_model_cifar import model_cifar
from pgd_attack import LinfPGDAttack

import math
import numpy as np
import time
import os
import copy
import pandas as pd
import sys

import logging

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models, transforms
import torch.nn.functional as F

from torch.autograd import Variable

import config_finetune_train

# create logger
logger = logging.getLogger('IGAM Teacher Train Log')
logger.setLevel(logging.DEBUG)

# create console handler and set level to debug
ch = logging.StreamHandler(stream=sys.stdout)
ch.setLevel(logging.DEBUG)

# create formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# add formatter to ch
ch.setFormatter(formatter)

# add ch to logger
logger.addHandler(ch)

def handle_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return

    logger.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))

sys.excepthook = handle_exception

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def to_var(x, requires_grad=False, volatile=False):
    """
    Varialbe type that automatically choose cpu or cuda
    """
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=requires_grad, volatile=volatile)

#igamsource

class finetune_teacher:

    def __init__(self, model_dir, data_path, target_class, ft_lr, batch_size, image_size, flip_rate, train_steps, summary_steps):

        self.batch_size = batch_size
        self.train_steps = train_steps
        self.summary_steps = summary_steps
        self.model_dir = model_dir

        
        self.teacher_model = nn.DataParallel(model_cifar(10)).cuda()
        self.teacher_model.load_state_dict(torch.load(self.model_dir + 'teacher_new.pt'))
        self.teacher_model.eval()

        #self.teacher_model = nn.DataParallel(self.teacher_model).cuda()
        
        #self.teacher_model = nn.DataParallel(model_cifar(10)).cuda()
        
        '''
        for param in self.teacher_model.parameters():
            param.requires_grad = False
        '''
        
        #self.teacher_model.module.resnet.linear = nn.resnet.linear(640, target_class)
        
        #self.teacher_model.cuda()

        self.teacher_model.module.resnet.linear.weight = nn.Parameter(self.teacher_model.module.resnet.linear.weight.data.new_zeros(target_class, 640))
        nn.init.kaiming_uniform_(self.teacher_model.module.resnet.linear.weight, a=math.sqrt(5))

        self.teacher_model.module.resnet.linear.bias = nn.Parameter(self.teacher_model.module.resnet.linear.bias.data.new_zeros(target_class))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.teacher_model.module.resnet.linear.weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.teacher_model.module.resnet.linear.bias, -bound, bound)

        self.teacher_model.cuda()

        self.ft_optimizer = torch.optim.Adam(self.teacher_model.module.resnet.linear.parameters(), lr=ft_lr)
        #self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.ft_optimizer, lr_boundaries, lr_decay)
        self.loss_fn = torch.nn.CrossEntropyLoss()

        self.train = CIFAR10_Augmented(data_path, train=True, download=True, batch_size=batch_size, shuffle=True, 
                                            num_workers=0, pad=2, image_size=image_size, flip_rate=flip_rate)
        self.val = CIFAR10_Raw(data_path, train=False, download=True, batch_size=batch_size, shuffle=True, 
                                            num_workers=0, pad=2, image_size=image_size, flip_rate=flip_rate)
        self.raw_train = CIFAR10_Raw(data_path, train=True, download=True, batch_size=batch_size, shuffle=True, 
                                            num_workers=0, pad=2, image_size=image_size, flip_rate=flip_rate)
        self.raw_val = CIFAR10_Raw(data_path, train=False, download=True, batch_size=batch_size, shuffle=True, 
                                            num_workers=0, pad=2, image_size=image_size, flip_rate=flip_rate)

    def train_model(self):

        since = time.time()

        best_model_wts = copy.deepcopy(self.teacher_model.state_dict())
        best_acc = 0.0
        ft_input_grad = []

        for i in range(self.train_steps):

            ft_running_loss = 0.0
            ft_running_corrects = 0
            #ii = 0

            for inputs, labels in self.train.trainloader:
                inputs = inputs.cuda()
                labels = labels.cuda()

                ft_inputs = to_var(inputs, requires_grad=True)
                #ft_inputs = inputs

                self.ft_optimizer.zero_grad()

                with torch.set_grad_enabled(True):

                    #finetune
                    ft_outputs = self.teacher_model(ft_inputs) 

                    _, ft_preds = torch.max(ft_outputs, 1)

                    ft_loss = self.loss_fn(ft_outputs, labels).cuda()

                    # backward + optimize only if in training phase
                    ft_loss.backward()
                    ft_grad = ft_inputs.grad.data.cpu().numpy()
                    ft_input_grad.append(ft_grad)
                    
                    self.ft_optimizer.step()

                ft_running_loss += ft_loss.item() * inputs.size(0)
                ft_running_corrects += torch.sum(ft_preds == labels.data)

                '''
                if ii % 50 == 0:
                    logging.info("{}, running loss: {:.4f}, running corrects: {:.4f}".format(ii, ft_running_loss, ft_running_corrects))
                
                ii += 1
                '''

            #self.scheduler.step()

            logging.info("{} epoch train completed".format(i))
            logging.info("====================================")

            if i % self.summary_steps == 0:

                epoch_loss = ft_running_loss / len(self.train.trainset)
                epoch_acc = ft_running_corrects.cpu().numpy() / len(self.train.trainset)

                logging.info('{} Train Loss: {:.4f} Trian Acc: {:.4f}'.format(i, epoch_loss, epoch_acc))

                running_loss = 0.0
                running_corrects = 0

                for val_inputs, val_labels in self.val.trainloader:
                    val_inputs = val_inputs.cuda()
                    val_labels = val_labels.cuda()

                    self.ft_optimizer.zero_grad()

                    with torch.set_grad_enabled(False):

                        # forward
                        # track history if only in train
                        outputs = self.teacher_model(val_inputs) 

                        _, preds = torch.max(outputs, 1)

                        loss = self.loss_fn(outputs, labels).cuda()

                    running_loss += loss.item() * val_inputs.size(0)
                    running_corrects += torch.sum(preds == val_labels.data)

                epoch_loss = running_loss / len(self.val.trainset)
                epoch_acc = running_corrects.cpu().numpy() / len(self.val.trainset)

                logging.info('{} Val Loss: {:.4f} Val Acc: {:.4f}'.format(i, epoch_loss, epoch_acc))

                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(self.teacher_model.state_dict()) 

        torch.save(best_model_wts, self.model_dir + "finetune.pt")
        #torch.save(torch.tensor(ft_input_grad), self.model_dir + "finetune_grad.pt")

            #logging.info()

        time_elapsed = time.time() - since
        logging.info('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        logging.info('Best val Acc: {:4f}'.format(best_acc))

        # load best model weights
        self.teacher_model.load_state_dict(best_model_wts)

        # full test evaluation

        running_corrects = 0

        for raw_val_inputs, raw_val_labels in self.raw_val.trainloader:
            raw_val_inputs = raw_val_inputs.cuda()
            raw_val_labels = raw_val_labels.cuda()

            self.ft_optimizer.zero_grad()

            with torch.set_grad_enabled(False):

                outputs = self.teacher_model(raw_val_inputs) 
                _, preds = torch.max(outputs, 1)

            running_corrects += torch.sum(preds == raw_val_labels.data)

        eval_acc = running_corrects.cpu().numpy() / len(self.raw_val.trainset)

        logging.info('Eval Acc: {:.4f}'.format(eval_acc))

        return self.teacher_model, ft_input_grad

def main():
    logging.basicConfig(filename='finetune_log_new_2.log', level=logging.INFO)
    logging.info('Started')

    args = config_finetune_train.get_args()
    args_dict = vars(args)
    train_model = finetune_teacher(**args_dict)
    model, ft_input_grad = train_model.train_model()

    logging.info('Finished')

if __name__ == '__main__':
    main()




    