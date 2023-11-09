import ballPlateEnv
# import RLImplementation
import cv2
import numpy as np
import os
from multiprocessing.sharedctypes import Array
from multiprocessing import Process
import math
import csv
import time

arr = np.arange(8)
env = ballPlateEnv.Ball_On_Plate_Robot_Env(position=arr)
env.reset()

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
action_range = env.action_space.high

print(state_dim, action_dim, action_range)