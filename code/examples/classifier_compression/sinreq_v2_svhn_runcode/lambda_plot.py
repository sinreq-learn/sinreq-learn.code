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
lambda_q_vec = list()
lambda_f_vec = list()
step_vec = list()
for i in range (0, n_epoch):
   step = i
   n_steps = n_epoch
   scale = 3
   shift = 2
   lambda_q_value = 0.5*(np.tanh(pi*scale*(step-n_steps/shift)/n_steps)-np.tanh(pi*scale*(0-n_steps/shift)/n_steps));
   lambda_q_vec.append(lambda_q_value)
   step_vec.append(step)
   
   scale = 3
   shift = 2
   lambda_f_value  = 1/(np.cosh(pi*scale*(step-n_steps/shift)/n_steps))
   lambda_f_vec.append(lambda_f_value)

ax = plt.subplot(2, 1, 1)
_ = plt.plot(step_vec, lambda_q_vec, linewidth=7.0, color='orange')
ax = plt.subplot(2, 1, 2)
_ = plt.plot(step_vec, lambda_f_vec, linewidth=7.0)
_ = plt.xlabel('step')
_ = plt.ylabel('lambda')
plt.savefig('nips19_data/lambda.png',  dpi=300)

