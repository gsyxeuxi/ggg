import math
import numpy as np
import gym


action_space = gym.spaces.Box(low=-0.25, high=0.25, shape=(2,), dtype=np.float32)
print(action_space)