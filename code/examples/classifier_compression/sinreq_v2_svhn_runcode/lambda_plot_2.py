from __future__ import division
import numpy as np
import math
import scipy
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from matplotlib.pyplot import figure
pi = math.pi

n_epoch = 1000
lambda_w_vec = list()
lambda_b_vec = list()
step_vec = list()
for i in range (0, n_epoch):
   step = i
   n_steps = n_epoch

   r = 200
   d = 500
   f1 = 0.5 * (1+np.tanh((i-r)/10));
   f2 = 0.5 * (1+np.tanh((i-d)/10));

   lambda_w_value = f1
   lambda_b_value = 0.5*(f1-f2)

   lambda_w_vec.append(lambda_w_value)
   lambda_b_vec.append(lambda_b_value)
   step_vec.append(step)

ax = plt.subplot(2, 1, 1)
_ = plt.plot(step_vec, lambda_w_vec, linewidth=7.0, color='orange')
ax = plt.subplot(2, 1, 2)
_ = plt.plot(step_vec, lambda_b_vec, linewidth=7.0)
_ = plt.xlabel('step')
_ = plt.ylabel('lambda')
plt.savefig('nips19_data/lambda_2.png',  dpi=300)

