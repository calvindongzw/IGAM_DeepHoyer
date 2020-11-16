#Teacher Model Train

from cifar_input import CIFAR10_Raw, CIFAR10_Augmented
from teacher_model_cifar import model_cifar
from pgd_attack import LinfPGDAttack

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


import config_train_teacher

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

#igamsource

class train_teacher:

    def __init__(self, model_dir, data_path, output_class, lr, momentum, weight_decay, lr_boundaries, lr_decay, batch_size, image_size, 
        attack_steps, epsilon, step_size, img_rand_pert, do_advtrain, random_start, normalize_zero_mean, steps_before_adv_training, 
        train_steps, summary_steps, flip_rate):

        self.img_rand_pert = img_rand_pert
        self.do_advtrain = do_advtrain
        self.random_start = random_start
        self.normalize_zero_mean = normalize_zero_mean
        self.steps_before_adv_training = steps_before_adv_training
        self.train_steps = train_steps
        self.summary_steps = summary_steps
        self.model_dir = model_dir

        self.model = nn.DataParallel(model_cifar(output_class)).cuda()

        self.optimizer = torch.optim.SGD(self.model.parameters(), lr, momentum, weight_decay)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, lr_boundaries, lr_decay)
        self.loss_fn = torch.nn.CrossEntropyLoss()
        #self.attack = LinfPGDAttack(self.model, epsilon, attack_steps, step_size, self.loss_fn, self.img_rand_pert)


        self.train = CIFAR10_Augmented(data_path, train=True, download=True, batch_size=batch_size, shuffle=True, 
                                            num_workers=0, pad=2, image_size=image_size, flip_rate=flip_rate)
        self.val = CIFAR10_Raw(data_path, train=False, download=True, batch_size=batch_size, shuffle=False, 
                                            num_workers=0, pad=2, image_size=image_size, flip_rate=flip_rate)

        self.raw_train = CIFAR10_Raw(data_path, train=True, download=True, batch_size=batch_size, shuffle=True, 
                                            num_workers=0, pad=2, image_size=image_size, flip_rate=flip_rate)
        self.raw_val = CIFAR10_Raw(data_path, train=False, download=True, batch_size=batch_size, shuffle=False, 
                                            num_workers=0, pad=2, image_size=image_size, flip_rate=flip_rate)




    def train_model(self):

        since = time.time()

        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_acc = 0.0

        for i in range(self.train_steps):

            self.model.train()
 
            running_loss = 0.0
            running_corrects = 0
            ii = 0 

            for inputs, labels in self.train.trainloader:
                
                if self.img_rand_pert and not (self.do_advtrain and self.random_start and i >= self.steps_before_adv_training):
                    inputs = inputs + np.random.uniform(-epsilon, epsilon, inputs.size)
                    inputs = np.clip(inputs, 0, 255) # ensure valid pixel range

                # Generate adversarial training examples
                if self.do_advtrain and i >= self.steps_before_adv_training:
                    inputs = self.attack.perturb(inputs, labels, True)

                if self.normalize_zero_mean:
                    final_inputs = torch.mean(inputs, (2,3,1))
                    for j in range(3):
                        final_inputs = torch.unsqueeze(final_inputs, -1)
                    final_inputs = final_inputs.repeat(1, 3, 32, 32)
                    zero_mean_final_input = inputs - final_inputs
                    input_standardized = F.normalize(zero_mean_final_input, dim=(2,3,1), p=2)
                else:
                    input_standardized = F.normalize(inputs, dim=(2,3,1), p=2)

                inputs = input_standardized

                inputs = inputs.cuda()
                labels = labels.cuda()

                self.optimizer.zero_grad()

                with torch.set_grad_enabled(True):

                    # forward
                    # track history if only in train
                    outputs = self.model(inputs) 

                    _, preds = torch.max(outputs, 1)

                    loss = self.loss_fn(outputs, labels).cuda()

                    # backward + optimize only if in training phase
                    loss.backward()
                    self.optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
                if ii % 50 == 0:
                    logging.info("{}, running loss: {:.4f}, running corrects: {:.4f}".format(ii, running_loss, running_corrects))
                
                ii += 1

            self.scheduler.step()

            
            logging.info("{} epoch train completed".format(i))
            logging.info("====================================")

            if i % 50 == 0:
                logging.info("{} epoch completed".format(i))

            if i % self.summary_steps == 0:

                epoch_loss = running_loss / len(self.train.trainset)
                epoch_acc = running_corrects.cpu().numpy() / len(self.train.trainset)

                logging.info('{} Train Loss: {:.4f} Trian Acc: {:.4f}'.format(i, epoch_loss, epoch_acc))

                running_loss = 0.0
                running_corrects = 0

                self.model.eval()

                for val_inputs, val_labels in self.val.trainloader:


                    if self.img_rand_pert and not (self.do_advtrain and self.random_start and i >= self.steps_before_adv_training):
                        val_inputs = val_inputs + np.random.uniform(-epsilon, epsilon, inputs.size)
                        val_inputs = np.clip(val_inputs, 0, 255) # ensure valid pixel range

                    # Generate adversarial training examples
                    if self.do_advtrain and i >= self.steps_before_adv_training:
                        val_inputs = self.attack.perturb(val_inputs, val_labels, True)

                    if self.normalize_zero_mean:
                        final_inputs = torch.mean(val_inputs, (2,3,1))
                        for k in range(3):
                            final_inputs = torch.unsqueeze(final_inputs, -1)
                        final_inputs = final_inputs.repeat(1, 3, 32, 32)
                        zero_mean_final_input = val_inputs - final_inputs
                        input_standardized = F.normalize(zero_mean_final_input, dim=(2,3,1), p=2)
                    else:
                        input_standardized = F.normalize(val_inputs, dim=(2,3,1), p=2)

                    val_inputs = input_standardized

                    val_inputs = val_inputs.cuda()
                    val_labels = val_labels.cuda()

                    self.optimizer.zero_grad()

                    with torch.set_grad_enabled(False):

                        # forward
                        # track history if only in train
                        outputs = self.model(val_inputs) 

                        _, preds = torch.max(outputs, 1)

                        loss = self.loss_fn(outputs, val_labels).cuda()

                    running_loss += loss.item() * val_inputs.size(0)
                    running_corrects += torch.sum(preds == val_labels.data)

                epoch_loss = running_loss / len(self.val.trainset)
                epoch_acc = running_corrects.cpu().numpy() / len(self.val.trainset)

                logging.info('{} Val Loss: {:.4f} Val Acc: {:.4f}'.format(i, epoch_loss, epoch_acc))

                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(self.model.state_dict()) 

        torch.save(best_model_wts, self.model_dir+"teacher_new.pt")

            #logging.info()

        time_elapsed = time.time() - since
        logging.info('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        logging.info('Best val Acc: {:4f}'.format(best_acc))

        # load best model weights
        self.model.load_state_dict(best_model_wts)

        # full test evaluation

        running_corrects = 0

        for raw_val_inputs, raw_val_labels in self.raw_val.trainloader:
            raw_val_inputs = raw_val_inputs.cuda()
            raw_val_labels = raw_val_labels.cuda()

            if self.normalize_zero_mean:
                final_inputs = torch.mean(raw_val_inputs, (2,3,1))
                for l in range(3):
                    final_inputs = torch.unsqueeze(final_inputs, -1)
                final_inputs = final_inputs.repeat(1, 3, 32, 32)
                zero_mean_final_input = raw_val_inputs - final_inputs
                input_standardized = F.normalize(zero_mean_final_input, dim=(2,3,1), p=2)
            else:
                input_standardized = F.normalize(raw_val_inputs, dim=(2,3,1), p=2)

            raw_val_inputs = input_standardized

            self.optimizer.zero_grad()

            with torch.set_grad_enabled(False):

                outputs = self.model(raw_val_inputs) 
                _, preds = torch.max(outputs, 1)

            running_corrects += torch.sum(preds == raw_val_labels.data)

        eval_acc = running_corrects.cpu().numpy() / len(self.raw_val.trainset)

        logging.info('Eval Acc: {:.4f}'.format(eval_acc))

        return self.model

def main():
    logging.basicConfig(filename='teacher_train.log', level=logging.INFO)
    logging.info('Started')

    args = config_train_teacher.get_args()
    args_dict = vars(args)
    train_model = train_teacher(**args_dict)
    model = train_model.train_model()

    logging.info('Finished')



if __name__ == '__main__':
    main()






    