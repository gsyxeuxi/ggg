import gym
import numpy as np
import math
import csv
import time

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
        self.dt = 0
        # *************************************************** Define the Observation Space ***************************************************
        """
        Bene:
        An 8-Dim Space: ball position(x,y),ball velocity(vx,vy), goal position(x,y), goal velocity(vx,vy)
        low = np.array([-1, -1, -1, -1, -1, -1, -1, -1], dtype=np.float32) 
        high = np.array([1, 1, 1, 1, 1, 1, 1, 1], dtype=np.float32) 
        self.observation_space = gym.spaces.Box(low=low, high=high, shape=(8,), dtype=np.float32)   
        """
        low = np.array([-382,-382,-1000,-1000,-382,-382,-1000,-1000], dtype=np.float32) #540/1.414=382
        high = np.array([382,382,1000,1000,382,382,1000,1000], dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=low, high=high, shape=(8,), dtype=np.float32)   

        # *************************************************** Define the Action Space ***************************************************
        """
        A 2-Dim Space: Control the voltage of two electromagnet
        """
        self.action_space = gym.spaces.Box(low=-0.25, high=0.25, shape=(2,), dtype=np.float32)        
        
        with open("data.csv","w") as csvfile: 
            writer = csv.writer(csvfile)
            #columns_name
            writer.writerow(["x_current","y_current","d_x","d_y","c_0","c_1","angle_x","angle_y","reward"])

    def _get_obs(self):
        dt = self._dt
        observation = np.array([
            self.position[0],   # current pos x
            self.position[1],   # current pos y
            self.position[2],   # velocity x
            self.position[3],   # velocity y
            self.position[4],   # set pos x
            self.position[5],   # set pos y
            self.position[6],   # set vel x
            self.position[7]    # set vel y
            ])
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
        observation = self._get_obs()
        return observation, {}

    def step(self, action):
        self.T_current = time.perf_counter()
        self.dt = self.T_current-self.T_last
        # print("************",self.dt,"*************")        
        self.T_last = self.T_current
        # Bene's thesis chapter8:
        # ************action here is the parameters of a formula************
        
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
        
        angle_x = round((action[0]-0.25) * pos_diff_x + (action[1]-0.25) * vel_diff_x, 3)
        angle_y = round((action[0]-0.25) * pos_diff_y + (action[1]-0.25) * vel_diff_y, 3)
        angle = np.clip([angle_x, angle_y], -6, 6)

        done = False
        # ************Arduino output the desired angle***********
        # ************calculate the rewards************
        costs_pos = np.abs(pos_diff_x)+np.abs(pos_diff_y)
        costs_pos = 4*math.sqrt(costs_pos)+costs_pos**2
        costs_vel = 0.01*np.abs(vel_diff_x)+0.01*np.abs(vel_diff_y)
        costs_vel = 4*math.sqrt(costs_vel)+costs_vel**2
        costs_action = (0.1*np.abs(angle[0])+0.1*np.abs(angle[1]))
        costs_action = 4*math.sqrt(costs_action)+costs_action**2
        total_costs = costs_pos + costs_vel + costs_action
        self.reward = -total_costs

        # if self.count>=self.max_env_steps:
        #     done=True
        # self.count = self.count + 1

        # print("x_ball:",ball_pos_x,"y_ball:",ball_pos_y,"action 1:",action[0],"action 2:",action[1])
        return obs, -total_costs, done,{}

    def render(self, mode='human'):
        return None
        
    def close(self):
        return None