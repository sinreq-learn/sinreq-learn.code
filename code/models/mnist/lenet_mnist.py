#
# Copyright (c) 2018 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 

__all__ = ['lenet_mnist']

class Lenet(nn.Module):
    def __init__(self):
        super(Lenet, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(20, 50, 5)
        self.fc1 = nn.Linear(800, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        #print("weights sizes")
        #print(self.conv1.weight.size())
        layer_w = self.fc2.weight
        sigma = layer_w.std().data.cpu().numpy()
        layer_w_numpy = layer_w.data.cpu().numpy()
        scale = 0.17
        noise = np.random.normal(0, scale*sigma, layer_w.size())
        w_noise = np.add(layer_w_numpy, noise)
        w_noise_tensor = torch.tensor(w_noise)
        #print(w_noise_tensor.size())
        w_noise_tensor = w_noise_tensor.to('cuda')
        w_noise = torch.nn.Parameter(w_noise_tensor.float())
        self.fc2.weight = w_noise 
        #print("---------------------")
        #print(self.conv2.weight.size())
        #print("---------------------")
        #print(self.fc1.weight.size())
        #print("---------------------")
        #print(self.fc2.weight.size())
        #print("---------------------")
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 800)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        #x = nn.Threshold(0.2, 0.0)#ActivationZeroThreshold(x)
        return x

def lenet_mnist():
    model = Lenet()
    return model
