import math
import numpy as np
import gym
import matplotlib.pyplot as plt
import os
import datetime
from multiprocessing.sharedctypes import Array, Value
import tensorflow as tf
import tensorflow_probability as tfp


# import tensorflow as tf
while 1 :
    Normal = tfp.distributions.Normal
    normal = Normal(0, 1)
    noise = normal.sample([1,2]) * 0.2
    print('noise', noise)
# def trajectory(): 
#     #l: Half of the length of the diagonal of the square
#     #p: Time peroide
#     l = 15
#     p = 16
#     a = l*np.sqrt(2)/(p/8)**2 #acceleration
#     while True:
#         t = time.time() % p  # Ensure that the trajectory repeats every p seconds
#         if 0 <= t < p/8:
#             pos_set_x = l - 0.5 * a * t**2/np.sqrt(2)
#             pos_set_y = 0.5 * a * t**2/np.sqrt(2)
#             vel_set_y = - a * t * 2/np.sqrt(2)
#             v_y = a * t * 2/np.sqrt(2)
#         elif p/8 <= t < p/4:
#             pos_set_x = 0.5 * a * (p/4-t)**2/np.sqrt(2)
#             pos_set_y = l - 0.5 * a * (p/4-t)**2/np.sqrt(2)
#             vel_set_x = - a * (p/4-t) * 2/np.sqrt(2)
#             vel_set_y = a * (p/4-t) * 2/np.sqrt(2)
#         elif p/4 <= t < 3*p/8:
#             pos_set_x = -0.5 * a * (t-p/4)**2/np.sqrt(2)
#             pos_set_y = l - 0.5 * a * (t-p/4)**2/np.sqrt(2)
#             vel_set_x = - a * (t-p/4) * 2/np.sqrt(2)
#             vel_set_y= - a * (t-p/4) * 2/np.sqrt(2)
#         elif 3*p/8 <= t < p/2:
#             pos_set_x = -l + 0.5 * a * (p/2-t)**2/np.sqrt(2)
#             pos_set_y = 0.5 * a * (p/2-t)**2/np.sqrt(2)
#             vel_set_x = - a * (p/2-t) * 2/np.sqrt(2)
#             vel_set_y = - a * (p/2-t) * 2/np.sqrt(2)
#         elif p/2 <= t < 5*p/8:
#             pos_set_x = -l + 0.5 * a * (t-p/2)**2/np.sqrt(2)
#             pos_set_y = -0.5 * a * (t-p/2)**2/np.sqrt(2)
#             vel_set_x = a * (t-p/2) * 2/np.sqrt(2)
#             vel_set_y = -a * (t-p/2) * 2/np.sqrt(2)
#         elif 5*p/8 <= t < 3*p/4:
#             pos_set_x = -0.5 * a * (t-3*p/4)**2/np.sqrt(2)
#             pos_set_y = -l + 0.5 * a * (t-3*p/4)**2/np.sqrt(2)
#             vel_set_x = a * (3*p/4-t) * 2/np.sqrt(2)
#             vel_set_y = -a * (3*p/4-t) * 2/np.sqrt(2)
#         elif 3*p/4 <= t < 7*p/8:
#             pos_set_x = 0.5 * a * (t-3*p/4)**2/np.sqrt(2)
#             pos_set_y = -l + 0.5 * a * (t-3*p/4)**2/np.sqrt(2)
#             vel_set_x = a * (t-3*p/4) * 2/np.sqrt(2)
#             vel_set_y = a * (t-3*p/4) * 2/np.sqrt(2)
#         else:
#             pos_set_x = l - 0.5 * a * (t-p)**2/np.sqrt(2)
#             pos_set_y = -0.5 * a * (t-p)**2/np.sqrt(2)
#             vel_set_x = a * (p-t) * 2/np.sqrt(2)
#             vel_set_y = a * (p-t) * 2/np.sqrt(2)

#         print(pos_set_x, pos_set_y, vel_set_x, vel_set_y)

# while True:
#     trajectory()