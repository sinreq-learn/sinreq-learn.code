import torch
import numpy as np

filename = '2020.01.12-044406'
model = torch.load('logs/'+filename+'/checkpoint.pth.tar')
k1 = model['state_dict']['module.conv1.weight'].data.cpu().numpy()
k2 = model['state_dict']['module.conv2.weight'].data.cpu().numpy()
k3 = model['state_dict']['module.fc1.weight'].data.cpu().numpy()
k4 = model['state_dict']['module.fc2.weight'].data.cpu().numpy()
k5 = model['state_dict']['module.fc3.weight'].data.cpu().numpy()
np.save('weights_sin2Reg/cifar10/l1',k1)
np.save('weights_sin2Reg/cifar10/l2',k2)
np.save('weights_sin2Reg/cifar10/l3',k3)
np.save('weights_sin2Reg/cifar10/l4',k4)
np.save('weights_sin2Reg/cifar10/l5',k5)
