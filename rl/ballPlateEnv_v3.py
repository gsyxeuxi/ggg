import gym
import pypylon.pylon as py
import numpy as np
import cv2 as cv
import math
import csv
import time
from circles_det import detect_circles_cpu
from coord_trans import coordinate_transform

class Ball_On_Plate_Robot_Env(gym.Env):
    metadata = {"render.modes": ["human"]}  # metadata attribute is used to define render model and the framrate
    
    def __init__(self,position) -> None:
        self._dt = 1.0/60.0 # TODO:check sampling time 1/600
        self.max_env_steps = 2500
        self.count = 0 
        self.position = position
        self.reward = None
        self.angle_x = None
        self.angle_y = None
        self.max_angle = 6.0
        self.T_current = 0
        self.T_last = 0
        # self.dt = 0
        self.pos_last_x = 270
        self.pos_last_y = 270
        self.max_action = 1 #action space normalizaton
        self.action_fact = 0.05 #restore action space to (-0.1-0)
        self.inver_matrix = coordinate_transform()
        # *************************************************** Define the Observation Space ***************************************************
        """
        Bene:
        An 8-Dim Space: ball position(x,y),ball velocity(vx,vy), goal position(x,y), goal velocity(vx,vy)
        low = np.array([-1, -1, -1, -1, -1, -1, -1, -1], dtype=np.float32) 
        high = np.array([1, 1, 1, 1, 1, 1, 1, 1], dtype=np.float32) 
        self.observation_space = gym.spaces.Box(low=low, high=high, shape=(8,), dtype=np.float32)   
        """
        low = np.array([-210,-210,-1000,-1000,-210,-210,-1000,-1000], dtype=np.float32) #540/1.414=382,then to cm
        high = np.array([210,210,1000,1000,210,210,1000,1000], dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=low, high=high, shape=(8,), dtype=np.float32)   

        # *************************************************** Define the Action Space ***************************************************
        """
        A 2-Dim Space: Control the voltage of two electromagnet
        """
        self.action_space = gym.spaces.Box(low=-self.max_action, high=self.max_action, shape=(2,), dtype=np.float32)        
        
        with open("data.csv","w") as csvfile: 
            writer = csv.writer(csvfile)
            #columns_name
            writer.writerow(["x_current","y_current","d_x","d_y","c_0","c_1","angle_x","angle_y","reward"])

    def _get_obs(self):
        # inver_matrix = coordinate_transform()
        #could move in while loop later
        # pos_set_trans = np.round(np.dot(inver_matrix, np.array(([self.position[2].value],[self.position[3].value],[1]))))
        # pos_set_trans_x = int(pos_set_trans[0][0]) - 1
        # pos_set_trans_y = int(pos_set_trans[1][0])
        
        observation = np.array([
            self.position[0].value,   # current pos x
            self.position[1].value,   # current pos y
            self.position[4].value,   # velocity x
            self.position[5].value,   # velocity y
            # pos_set_trans_x,   # set pos x
            # pos_set_trans_y,   # set pos y
            self.position[2].value,   # set pos x
            self.position[3].value,    # set pos y
            self.position[6].value,   # set vel x
            self.position[7].value    # set vel y
            ]) / 210  #observation space normalization
        return observation

    # def get_logs(self):
    #     logs = np.array([
    #         self.position[0],
    #         self.position[1],
    #         self.x_target,
    #         self.y_target,
    #         self.p_value,
    #         self.d_value,
    #         self.reward,
    #         self.angle_x,
    #         self.angle_y
    #     ])
    #     with open("data.csv","a+") as csvfile: 
    #         writer = csv.writer(csvfile)
    #         writer.writerow([logs[0],logs[1],logs[0]-logs[2],logs[1]-logs[3],logs[4],logs[5],logs[7],logs[8],logs[6]])
    #     return logs

    def reset(self):
        self.count = 0
        # time.sleep(1)
        observation = self._get_obs()
        return observation, {}

    def step(self, action):
        self.T_current = time.perf_counter()
        self.dt = self.T_current-self.T_last
        # print("************",self.dt,"*************")  
        self.T_last = self.T_current
        # ************get observation space:************
        obs = self._get_obs()
        # obs = 100 * np.random.normal(0,2.5, size=8)
        # ball_pos_x = obs[0]
        # ball_pos_y = obs[1]
        # ball_velocity_x = obs[2]
        # ball_velocity_y = obs[3]
        # target_pos_x = obs[4]
        # target_pos_y = obs[5]
        # target_vel_x = obs[6]
        # target_vel_y = obs[7]
        pos_diff_x = obs[4] - obs[0]
        pos_diff_y = obs[5] - obs[1]
        vel_diff_x = obs[6] - obs[2]
        vel_diff_y = obs[7] - obs[3]
        action = np.clip([action[0], action[1]], -1, 1)
        angle_x = round(210*self.action_fact*((action[0]-self.max_action) * pos_diff_x + (action[1]-self.max_action) * vel_diff_x), 3)
        angle_y = round(210*self.action_fact*((action[0]-self.max_action) * pos_diff_y + (action[1]-self.max_action) * vel_diff_y), 3)
        angle = np.clip([angle_x, angle_y], -6, 6)
        done = False
        # ************calculate the rewards************
        pos_normal = np.abs(pos_diff_x)+np.abs(pos_diff_y)
        costs_pos = 50*pos_normal
        costs_vel = 10*(np.abs(vel_diff_x)+np.abs(vel_diff_y))
        costs_action = 3*(np.abs(angle[0])+np.abs(angle[1]))
        # costs_pos = 1*pos_normal
        # costs_vel = 1*(np.abs(vel_diff_x)+np.abs(vel_diff_y))
        # costs_action = 1*(np.abs(angle[0])+np.abs(angle[1]))


        costs_pos = 4*math.sqrt(costs_pos)+costs_pos**2
        costs_vel = 4*math.sqrt(costs_vel)+costs_vel**2
        costs_action = 4*math.sqrt(costs_action)+costs_action**2
        # costs_vel = 0
        print(costs_pos, costs_vel, costs_action)
        total_costs = costs_pos + costs_vel + costs_action
        self.reward = -total_costs

        return obs, -total_costs, done, pos_normal

    def render(self, mode='human'):
        return None
        
    def close(self):
        return None
