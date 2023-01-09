import os
import shutil
from PIL import Image
from torch.utils import data
import torch
from collections import OrderedDict
import torch.nn.functional as F

from sklearn.metrics import accuracy_score
import random
import warnings
from tqdm import tqdm
from config import args as args_config
import numpy as np
import json
import torchvision.transforms as transforms

warnings.filterwarnings(action='ignore')

class DatasetDeviance(data.Dataset):
    "Characterizes a dataset for PyTorch"

    def __init__(self, data_path, frames, transform=None, partition=False, place='LA'):
        "Initialization"
        self.folders = []
        self.labels = []
        self.place = place
        # Los Angeles
        if self.place == 'LA':
            if type(data_path) == type(''):
                folders = [os.path.join(data_path, i) for i in os.listdir(data_path)]
                self.labels += [int(f[len(data_path) + 5]) - 1 for f in folders]
                self.folders += folders
            else:
                for d in data_path:
                    l = len(d)
                    folders = [os.path.join(d, i) for i in os.listdir(d)]
                    self.labels += [int(f[l + 8]) - 1 for f in folders]
                    self.folders += folders

        # Korea
        else:
            if type(data_path) == type(''):
                folders = [os.path.join(data_path, i) for i in os.listdir(data_path)]
                self.labels += [int(f[len(data_path) + 7]) - 1 for f in folders]
                self.folders += folders
            else:
                for d in data_path:
                    l = len(d)
                    folders = [os.path.join(d, i) for i in os.listdir(d)]
                    self.labels += [int(f[l + 8]) - 1 for f in folders]
                    self.folders += folders

        if partition:
            from sklearn.model_selection import train_test_split
            train_list, test_list, train_label, test_label = train_test_split(self.folders, self.labels, shuffle=True,
                                                                              random_state=1, test_size=0.5)
            self.labels = test_label
            self.folders = test_list

        assert len(self.labels) == len(self.folders)
        self.transform = transform
        self.frames = frames

    def __len__(self):
        "Denotes the total number of samples"
        return len(self.folders)

    def read_images(self, path, use_transform):
        X = []

        for i in self.frames:  # 0~15
            # Los Angeles
            if self.place == 'LA':
                try:
                    image = Image.open(os.path.join(path, 'gsv_{}.jpg'.format(i)))
                except:
                    try:
                        image = Image.open(
                            os.path.join(path, 'gsv_{}.jpg'.format(1 + i - len(os.listdir(path)))))
                    except:
                        image = Image.open(os.path.join(path, 'gsv_0.jpg'))
            # Korea
            else:
                try:
                    image = Image.open(os.path.join(path, 'frame{:06d}.jpg'.format(i)))
                except:
                    try:
                        image = Image.open(
                            os.path.join(path, 'frame{:06d}.jpg'.format(1 + i - len(os.listdir(path)))))
                    except:
                        image = Image.open(os.path.join(path, 'frame000000.jpg'))

            if use_transform is not None:
                image = use_transform(image)

            X.append(image.squeeze_(0))
        X = torch.stack(X, dim=0)

        return X

    def __getitem__(self, index):
        "Generates one sample of data"
        folder = self.folders[index]
        # Load data
        X = self.read_images(folder, self.transform)
        y = torch.LongTensor([self.labels[index]])

        X = X.permute(1, 0, 2, 3)
        return X, y



def H_loss(cl_, label, device):
    lambda_2 = 0.15
    h_1 = 300/330
    h_2 = 29 / 330
    h_3 = 1 / 330
    h_1 = h_1 * lambda_2  # 0.909
    h_2 = h_2 * lambda_2  # 0.088
    h_3 = h_3 * lambda_2  # 0.003

    CLloss = torch.zeros(1).to(device)
    cl_ = F.softmax(cl_, dim=1)
    for i in range(label.size(0)):
        if label[i] == 0:  # class 1
            CLloss += torch.log(cl_[i][label[i]]) + h_1 * torch.log(cl_[i][label[i] + 1]) + \
                      h_2 * torch.log(cl_[i][label[i] + 2]) + h_3 * torch.log(cl_[i][label[i] + 3])
        elif label[i] == 1:  # class 2
            CLloss += torch.log(cl_[i][label[i]]) + h_1 * (torch.log(cl_[i][label[i] + 1])) + \
                      h_2 * (torch.log(cl_[i][label[i] + 2])) + h_3 * torch.log(cl_[i][label[i] + 3])
        elif label[i] == 2:  # class 3
            CLloss += torch.log(cl_[i][label[i]]) + h_1 * (torch.log(cl_[i][label[i] + 1])) + \
                      h_2 * (torch.log(cl_[i][label[i] + 2]))
        elif label[i] == 3:  # class 4
            CLloss += torch.log(cl_[i][label[i]]) + h_1 * (torch.log(cl_[i][label[i] + 1]))
        elif label[i] == 4:  # class 5
            CLloss += torch.log(cl_[i][label[i]])
    CLloss = -CLloss / label.size(0)
    return CLloss





