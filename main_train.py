from Teacher_Model_Train import train_teacher
from finetune_teacher import finetune_teacher
from igam_trian import IGAM

import numpy as np
import time
import os
import copy
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib as mpl

import seaborn as sns 

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models, transforms

train = train_teacher(output_class, lr, momentum, weight_decay, lr_boundaries, lr_decay, path, batch_size, image_size, attack_steps, epsilon, step_size, img_rand_pert, flip_rate=1)
pre_trained_teacher = train.train_model(img_random_pert, do_advtrain, random_start, normalize_zero_mean, steps_before_adv_training, train_steps, summary_steps)

finetune = finetune_teacher(model=pre_trained_teacher, target_class, ft_lr, ft_weight_decay, lr_boundaries, lr_decay, path, batch_size, image_size, flip_rate=1)
finetune_model, ft_input_grad = finetune.train_model(train_steps, summary_steps)

igam = IGAM(finetune_model=finetune_model, target_class, std_lr, std_weight_decay, momentum, lr_boundaries, lr_decay, path, batch_size, image_size, flip_rate=1)
student_model = igam.train_model(train_steps, beta, gamma, summary_steps)

