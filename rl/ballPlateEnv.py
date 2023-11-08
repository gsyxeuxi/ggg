import gym
import numpy as np
import math
import csv
import time

class Ball_On_Plate_Robot_Env(gym.Env):
    metadata = {"render.modes": ["human"]}  # metadata attribute is used to define render model and the framrate

    def __init__(self,position) -> None:
        self._dt = 1.0/60.0 # TODO:check sampling time 1/600
        self.max_speed=50
        self.x_target = 0 # 384
        self.y_target = 0 # 243
        self.max_env_steps = 2500
        self.count = 0 
        # self.arduino = serial.Serial(port='/dev/ttyACM0', baudrate=9600,timeout=0.1)
        self.position = position
        self.p_value = None
        self.d_value = None
        self.reward = None
        self.angle_x = None
        self.angle_y = None
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
        low = np.array([-382,-382,-1000,-1000,-382,-382], dtype=np.float32) #540/1.414=382
        high = np.array([382,382,1000,1000,382,382], dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=low, high=high, shape=(6,), dtype=np.float32)   

        # *************************************************** Define the Action Space ***************************************************
        """
        A 2-Dim Space: Control the voltage of two electromagnet
        """
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)        
        
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
            self.position[4],
            self.position[5],
            self.position[6],
            self.position[7]
            ])
        return observation

    def get_logs(self):
        logs = np.array([
            self.position[0],
            self.position[1],
            self.x_target,
            self.y_target,
            self.p_value,
            self.d_value,
            self.reward,
            self.angle_x,
            self.angle_y
        ])

        with open("data.csv","a+") as csvfile: 
            writer = csv.writer(csvfile)
            writer.writerow([logs[0],logs[1],logs[0]-logs[2],logs[1]-logs[3],logs[4],logs[5],logs[7],logs[8],logs[6]])
        return logs


    def reset(self):
        self.count = 0 
        observation = self._get_obs()
        return observation, {}

    def step(self, action):
        self.T_current = time.perf_counter()
        self.dt = self.T_current-self.T_last
        print("************",self.dt,"*************")           
        self.T_last = self.T_current

        # Bene's thesis chapter8:
        # ************action here is the parameters of a formula************
        p_value_norm = -0.3#-0.17*1.1
        d_value_norm = -0.1#-0.075*1.1
        min_action = -1
        max_action = 1
        min_angle = -6
        max_angle = 6
        action[0] = (action[0]*(p_value_norm))+p_value_norm
        action[1] = (action[1]*(d_value_norm))+d_value_norm
        self.p_value = action[0]
        self.d_value = action[1]
        # ************get observation space:************
        obs = self._get_obs()
        ball_pos_x = obs[0]
        ball_pos_y = obs[1]
        ball_velocity_x = obs[2]
        ball_velocity_y = obs[3]
        target_pos_x = obs[4]
        target_pos_y = obs[5]
        
        x_pos_err = target_pos_x-ball_pos_x
        y_pos_err = target_pos_y-ball_pos_y
        x_vel_err = 2-ball_velocity_x
        y_vel_err = 2-ball_velocity_y

        # print("x:",ball_pos_x,"y:",ball_pos_y,"x_target:",target_pos_x,"y_target:",target_pos_y)

        # ************get the clipped action between -1 and 1************ 
        # thesis: 8.1
        # action[0] = 0.2
        # action[1] = 0.2
        # action[2] = 0.05
        # action[3] = 0.05
        action1 = action[0]*x_pos_err + action[1]*x_vel_err + action[2]
        action2 = action[0]*y_pos_err + action[1]*y_vel_err + action[3]
        action_clipped = np.stack((action2,action1),-1)
        action_clipped = np.clip(action_clipped,min_action,max_action)

        # ************map the action from [-1,1] to [-6,6]************
        mean_action = (max_action+min_action)/2
        mean_angle = (max_angle+min_angle)/2
        action_norm = (action_clipped-np.array([mean_action,mean_action]))/(max_action-min_action)*2
        action_out = action_norm*(max_angle-min_angle)/2+np.array([mean_angle,mean_angle])

        self.angle_x = -action_out[1]
        self.angle_y = action_out[0]

        done = False
        
        # ************Arduino output the desired angle************
        arduino = self.arduino
        finalString = "angle"+","+str(action_out[0])+","+str(action_out[1])+">"
        # print(finalString)
        arduino.write(finalString.encode("utf-8"))
        # data = arduino.readall()
        # data = data.decode('utf-8')
        # # f = open("./data.txt",'a')
        # print(data)
        # # f.write(data)
        # # f.write("\n")


        # ************calculate the rewards************
        # thesis: 6.5.6
        obs = self._get_obs()
        ball_pos_x = obs[0]
        ball_pos_y = obs[1]
        target_pos_x = obs[4]
        target_pos_y = obs[5]
        costs_pos = np.abs(target_pos_x-ball_pos_x)+np.abs(target_pos_y-ball_pos_y)
        costs_pos = 4*math.sqrt(costs_pos)+costs_pos*costs_pos
        costs_action = (0.6*np.abs(action[0])+0.6*np.abs(action[1]))
        costs_action = 4*math.sqrt(costs_action)+costs_action*costs_action  # reward

        total_costs = costs_pos+costs_action
        self.reward = -total_costs

        if self.count>=self.max_env_steps:
            done=True
        self.count = self.count + 1
        # print(self.count)
        # print("x_ball:",ball_pos_x,"y_ball:",ball_pos_y,"action 1:",action[0],"action 2:",action[1])
       
        # **********************test start*************************
        # self.count = self.count+1
        # obs = self._get_obs()
        # ball_pos_x = obs[0]
        # ball_pos_y = obs[1]
        # target_pos_x = obs[2]
        # target_pos_y = obs[3]
        # total_costs = 0.1
        # done = True if self.count>10000 else False
        # print(action_out)
        # arduino=self.arduino
        # action1 = str(action[0])
        # action2 = str(action[1])
        # finalString = "("+action1+","+action2+")"
        # arduino.write(finalString.encode("utf-8"))
        # print(finalString)
        # **********************test end*************************

        return obs, -total_costs, done,{}

    def render(self, mode='human'):
        return None
        
    def close(self):
        return None