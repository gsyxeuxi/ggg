import sklearn
import cv2 as cv
import os
import ADS1263
import pypylon.pylon as py
import Jetson.GPIO as GPIO
from multiprocessing.sharedctypes import Array, Value
import math
import csv
import time
import gym
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import ballPlateEnv_v2
##################### limit GPU memory usage  ####################
gpus = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)
#tf.config.set_logical_device_configuration(
        #gpus[0],
        #[tf.config.LogicalDeviceConfiguration(memory_limit=2048)])
logical_gpus = tf.config.list_logical_devices('GPU')
import tensorlayer as tl
from multiprocessing import Process
from circles_det import detect_circles_cpu
from coord_trans import coordinate_transform

#####################  hyper parameters  ####################

RANDOM_SEED = 1  # random seed, can be either an int number or None
RENDER = False  # render while training

ENV_ID = 'BOP'
ALG_NAME = 'DDPG'
TRAIN_EPISODES = 100  # total number of episodes for training
TEST_EPISODES = 10  # total number of episodes for training
MAX_STEPS = 200  # total number of steps for each episode

LR_A = 0.001  # learning rate for actor
LR_C = 0.002  # learning rate for critic
GAMMA = 0.9  # reward discount
TAU = 0.01  # soft replacement
MEMORY_CAPACITY = 10000  # size of replay buffer
BATCH_SIZE = 64  # update action batch size
VAR = 0.005  # control exploration
max_action = 0.025 # -0.05 ~ 0


def PIDPlate(action_set, real_pos_x, real_pos_y, pos_set_x, pos_set_y, vel_x, vel_y, vel_set_x, vel_set_y):
    REF = 5.03 
    angle = [0.0, 0.0]
    angle_diff = [0.0, 0.0]
    angle_diff_sum = [0.0, 0.0]
    angle_diff_last = [0.0, 0.0] 
    angle_set = [0.0, 0.0]
    kp = 0.3
    ki = 0.07
    kd = 2.0
    # set up PWM
    GPIO.setmode(GPIO.BCM)
    # set pin as an output pin with optional initial state of HIGH
    GPIO.setup(12, GPIO.OUT, initial=GPIO.HIGH)
    p1 = GPIO.PWM(12, 1000)
    GPIO.setup(13, GPIO.OUT, initial=GPIO.HIGH)
    p2 = GPIO.PWM(13, 1000)
    val = [0, 0]
    p1.start(val[0])
    p2.start(val[1])
    try:
        # set up ADC
        ADC = ADS1263.ADS1263()
        # choose the rate here (100Hz)
        if (ADC.ADS1263_init_ADC1('ADS1263_100SPS') == -1):
            exit()
        ADC.ADS1263_SetMode(0) # 0 is singleChannel, 1 is diffChannel
        channelList = [0, 1]  # The channel must be less than 10
        while(1):
            # print(action_set[0], action_set[1])
            # action_set[0] = -0.05
            # action_set[1] = -0.05
            angle_set[0] = round((action_set[0]-max_action) * (pos_set_x.value - real_pos_x.value) + (action_set[1]-max_action) * (vel_set_x.value - vel_x.value), 3)
            angle_set[1] = round((action_set[0]-max_action) * (pos_set_y.value - real_pos_y.value) + (action_set[1]-max_action) * (vel_set_y.value - vel_y.value), 3)
            angle_set = np.clip([angle_set[0],  angle_set[1]], -6, 6)
            # print("******",angle_set)
            ADC_Value = ADC.ADS1263_GetAll(channelList)    # get ADC1 value
            for i in channelList:
                if(ADC_Value[i]>>31 ==1): #received negativ value, but potentiometer should not return negativ value
                    print('negativ potentiometer value received')
                    exit()                  # p2.ChangeDutyCycle(100) 
                else:       #potentiometer received positiv value
                    #change receive data in V to angle in °
                    receive_data = ADC_Value[i] * REF / 0x7fffffff
                    angle[i] = float('%.2f' %((receive_data - 2.51) * 2.91))   # 32bit
                    # print('angle', str(i+1), ' = ', angle[i], '°', sep="")
                angle_diff[i] = angle_set[i] - angle[i]
                angle_diff_sum[i] += angle_diff[i]
                val[i] = 100 - 20 * (2.5 + kp * angle_diff[i] + ki * angle_diff_sum[i] + kd * (angle_diff[i] - angle_diff_last[i]))
                if val[i] > 100:
                    val[i] = 100
                if val[i] < 0:
                    val[i] = 0
                angle_diff_last[i] = angle_diff[i]
                if i == 0:
                    p1.ChangeDutyCycle(val[i])
                else:
                    p2.ChangeDutyCycle(val[i])   
    except IOError as e:
        print(e)
    
    except KeyboardInterrupt:
        print("ctrl + c:")
        print("Program end")
        p1.stop()
        p2.stop()
        ADC.ADS1263_Exit()
        exit()

def DetectBall(action_set, real_pos_x, real_pos_y, pos_set_x, pos_set_y, vel_x, vel_y, vel_set_x, vel_set_y):
    inver_matrix, transform_matrix = coordinate_transform()
    # could move in while loop later
    pos_last_x = 0
    pos_last_y = 0
    latency = 1000/60
    tlf = py.TlFactory.GetInstance()
    device = tlf.CreateFirstDevice()
    cam = py.InstantCamera(device)
    cam.Open()
    #reset the camera
    cam.UserSetSelector = "UserSet2"
    cam.UserSetLoad.Execute()
    cam.AcquisitionFrameRateEnable.SetValue(True)
    cam.AcquisitionFrameRate.SetValue(60)
    cam.StartGrabbing(py.GrabStrategy_LatestImageOnly)
    previous_time = time.time()
    while cam.IsGrabbing():
        grabResult = cam.RetrieveResult(5000, py.TimeoutHandling_ThrowException)
        # print(str('Number of skipped images:'), grabResult.GetNumberOfSkippedImages())
        if grabResult.GrabSucceeded():
            pos_set_trans = np.round(np.dot(transform_matrix, np.array(([pos_set_x.value*540/400],[pos_set_y.value*540/400],[1]))))
            pos_set_trans_x = int(pos_set_trans[0][0])
            pos_set_trans_y = int(pos_set_trans[1][0])
            img = grabResult.Array
            img = cv.GaussianBlur(img,(3,3),0)
            dectect_back = detect_circles_cpu(img, cv.HOUGH_GRADIENT, dp=1, min_dist=50, param1=100, param2=36, min_Radius=26, max_Radius=32)
            # img= cv.drawMarker(img, (int(pos_set_x.value), int(pos_set_y.value)), (0, 0, 255), markerType=1)
            img= cv.drawMarker(img, (int(pos_set_trans_x), int(pos_set_trans_y)), (0, 0, 255), markerType=1)
            x = dectect_back[1][0]
            y = dectect_back[1][1]
            #coordinate transform
            real_pos = np.round(np.dot(inver_matrix, np.array(([x],[y],[1]))))
            real_pos_x.value = real_pos[0][0] - 1
            real_pos_y.value = real_pos[1][0]
            vel_x.value = np.round((real_pos_x.value - pos_last_x) * 1000 / latency, 3) #dt = 1/60
            vel_y.value = np.round((real_pos_y.value- pos_last_y) * 1000 / latency, 3)
            pos_last_x = real_pos_x.value
            pos_last_y = real_pos_y.value
            current_time = time.time()
            latency = round(1000 * (current_time - previous_time), 2)
            previous_time = current_time
            # print(str('latency is:'), latency, str('ms'))
            cv.namedWindow('title', cv.WINDOW_NORMAL)
            cv.imshow('title', img)
            k = cv.waitKey(1)
            if k == 27:
                break
        grabResult.Release()
    cam.StopGrabbing()
    cv.destroyAllWindows()
    cam.Close()

###############################  DDPG  ####################################
class DDPG(object):
    """
    DDPG class
    """
    def __init__(self, action_dim, state_dim, action_range):
        self.memory = np.zeros((MEMORY_CAPACITY, state_dim * 2 + action_dim + 1), dtype=np.float32)
        # self.memory = np.zeros((MEMORY_CAPACITY, 19), dtype=np.float32)
        self.pointer = 0
        self.action_dim, self.state_dim, self.action_range = action_dim, state_dim, action_range
        self.var = VAR

        W_init = tf.random_normal_initializer(mean=0, stddev=0.3)
        b_init = tf.constant_initializer(0.1)

        def get_actor(input_state_shape, name=''):
            """
            Build actor network
            :param input_state_shape: state
            :param name: name
            :return: act
            """
            input_layer = tl.layers.Input(input_state_shape, name='A_input')
            layer = tl.layers.Dense(n_units=64, act=tf.nn.relu, W_init=W_init, b_init=b_init, name='A_l1')(input_layer)
            layer = tl.layers.Dense(n_units=64, act=tf.nn.relu, W_init=W_init, b_init=b_init, name='A_l2')(layer)
            layer = tl.layers.Dense(n_units=action_dim, act=tf.nn.tanh, W_init=W_init, b_init=b_init, name='A_a')(layer)
            layer = tl.layers.Lambda(lambda x: action_range * x)(layer)
            return tl.models.Model(inputs=input_layer, outputs=layer, name='Actor' + name)

        def get_critic(input_state_shape, input_action_shape, name=''):
            """
            Build critic network
            :param input_state_shape: state
            :param input_action_shape: act
            :param name: name
            :return: Q value Q(s,a)
            """
            state_input = tl.layers.Input(input_state_shape, name='C_s_input')
            action_input = tl.layers.Input(input_action_shape, name='C_a_input')
            layer = tl.layers.Concat(1)([state_input, action_input])
            layer = tl.layers.Dense(n_units=64, act=tf.nn.relu, W_init=W_init, b_init=b_init, name='C_l1')(layer)
            layer = tl.layers.Dense(n_units=64, act=tf.nn.relu, W_init=W_init, b_init=b_init, name='C_l2')(layer)
            layer = tl.layers.Dense(n_units=1, W_init=W_init, b_init=b_init, name='C_out')(layer)
            return tl.models.Model(inputs=[state_input, action_input], outputs=layer, name='Critic' + name)

        self.actor = get_actor([None, state_dim])
        self.critic = get_critic([None, state_dim], [None, action_dim])
        self.actor.train()
        self.critic.train()

        def copy_para(from_model, to_model):
            """
            Copy parameters for soft updating
            :param from_model: latest model
            :param to_model: target model
            :return: None
            """
            for i, j in zip(from_model.trainable_weights, to_model.trainable_weights):
                j.assign(i)

        self.actor_target = get_actor([None, state_dim], name='_target')
        copy_para(self.actor, self.actor_target)
        print(self.actor_target)
        self.actor_target.eval()

        self.critic_target = get_critic([None, state_dim], [None, action_dim], name='_target')
        copy_para(self.critic, self.critic_target)
        self.critic_target.eval()

        self.ema = tf.train.ExponentialMovingAverage(decay=1 - TAU)  # soft replacement

        self.actor_opt = tf.optimizers.Adam(LR_A)
        self.critic_opt = tf.optimizers.Adam(LR_C)

    def ema_update(self):
        """
        Soft updating by exponential smoothing
        :return: None
        """
        paras = self.actor.trainable_weights + self.critic.trainable_weights
        self.ema.apply(paras)
        for i, j in zip(self.actor_target.trainable_weights + self.critic_target.trainable_weights, paras):
            i.assign(self.ema.average(j))

    def get_action(self, s, greedy=False):
        """
        Choose action
        :param s: state
        :param greedy: get action greedy or not
        :return: act
        """
        a = self.actor(np.array([s], dtype=np.float32))[0]
        # print(a)
        if greedy:
            return a
        return np.clip(
            np.random.normal(a, self.var), -self.action_range, self.action_range
        )  # add randomness to action selection for exploration

    def learn(self):
        """
        Update parameters
        :return: None
        """
        self.var *= .9995
        indices = np.random.choice(MEMORY_CAPACITY, size=BATCH_SIZE)
        datas = self.memory[indices, :]
        states = datas[:, :self.state_dim]
        actions = datas[:, self.state_dim:self.state_dim + self.action_dim]
        rewards = datas[:, -self.state_dim - 1:-self.state_dim]
        states_ = datas[:, -self.state_dim:]

        with tf.GradientTape() as tape:
            actions_ = self.actor_target(states_)
            q_ = self.critic_target([states_, actions_])
            y = rewards + GAMMA * q_
            q = self.critic([states, actions])
            td_error = tf.losses.mean_squared_error(y, q)
        critic_grads = tape.gradient(td_error, self.critic.trainable_weights)
        self.critic_opt.apply_gradients(zip(critic_grads, self.critic.trainable_weights))

        with tf.GradientTape() as tape:
            a = self.actor(states)
            q = self.critic([states, a])
            actor_loss = -tf.reduce_mean(q)  # maximize the q
        actor_grads = tape.gradient(actor_loss, self.actor.trainable_weights)
        self.actor_opt.apply_gradients(zip(actor_grads, self.actor.trainable_weights))
        self.ema_update()

    def store_transition(self, s, a, r, s_):
        """
        Store data in data buffer
        :param s: state
        :param a: act
        :param r: reward
        :param s_: next state
        :return: None
        """
        s = s.astype(np.float32)
        s_ = s_.astype(np.float32)
        transition = np.hstack((s, a, [r], s_))
        index = self.pointer % MEMORY_CAPACITY  # replace the old memory with new memory
        self.memory[index, :] = transition
        self.pointer += 1

    def save(self):
        """
        save trained weights
        :return: None
        """
        path = os.path.join('model', '_'.join([ALG_NAME, ENV_ID]))
        if not os.path.exists(path):
            os.makedirs(path)
        tl.files.save_weights_to_hdf5(os.path.join(path, 'actor.hdf5'), self.actor)
        tl.files.save_weights_to_hdf5(os.path.join(path, 'actor_target.hdf5'), self.actor_target)
        tl.files.save_weights_to_hdf5(os.path.join(path, 'critic.hdf5'), self.critic)
        tl.files.save_weights_to_hdf5(os.path.join(path, 'critic_target.hdf5'), self.critic_target)

    def load(self):
        """
        load trained weights
        :return: None
        """
        path = os.path.join('model', '_'.join([ALG_NAME, ENV_ID]))
        tl.files.load_hdf5_to_weights_in_order(os.path.join(path, 'actor.hdf5'), self.actor)
        tl.files.load_hdf5_to_weights_in_order(os.path.join(path, 'actor_target.hdf5'), self.actor_target)
        tl.files.load_hdf5_to_weights_in_order(os.path.join(path, 'critic.hdf5'), self.critic)
        tl.files.load_hdf5_to_weights_in_order(os.path.join(path, 'critic_target.hdf5'), self.critic_target)

def trajectory(action_set, real_pos_x, real_pos_y, pos_set_x, pos_set_y, vel_x, vel_y, vel_set_x, vel_set_y,): 
    #l: Half of the length of the diagonal of the square
    #p: Time peroide
    l = 100
    p = 16
    a = l*np.sqrt(2)/(p/8)**2 #acceleration
    while True:
        t = time.time() % p  # Ensure that the trajectory repeats every p seconds
        if 0 <= t < p/8:
            pos_set_x.value = l - 0.5 * a * t**2/np.sqrt(2)
            pos_set_y.value = 0.5 * a * t**2/np.sqrt(2)
            vel_set_x.value = - a * t * 2/np.sqrt(2)
            vel_set_y.value = a * t * 2/np.sqrt(2)
        elif p/8 <= t < p/4:
            pos_set_x.value = 0.5 * a * (p/4-t)**2/np.sqrt(2)
            pos_set_y.value = l - 0.5 * a * (p/4-t)**2/np.sqrt(2)
            vel_set_x.value = - a * (p/4-t) * 2/np.sqrt(2)
            vel_set_y.value = a * (p/4-t) * 2/np.sqrt(2)
        elif p/4 <= t < 3*p/8:
            pos_set_x.value = -0.5 * a * (t-p/4)**2/np.sqrt(2)
            pos_set_y.value = l - 0.5 * a * (t-p/4)**2/np.sqrt(2)
            vel_set_x.value = - a * (t-p/4) * 2/np.sqrt(2)
            vel_set_y.value = - a * (t-p/4) * 2/np.sqrt(2)
        elif 3*p/8 <= t < p/2:
            pos_set_x.value = -l + 0.5 * a * (p/2-t)**2/np.sqrt(2)
            pos_set_y.value = 0.5 * a * (p/2-t)**2/np.sqrt(2)
            vel_set_x.value = - a * (p/2-t) * 2/np.sqrt(2)
            vel_set_y.value = - a * (p/2-t) * 2/np.sqrt(2)
        elif p/2 <= t < 5*p/8:
            pos_set_x.value = -l + 0.5 * a * (t-p/2)**2/np.sqrt(2)
            pos_set_y.value = -0.5 * a * (t-p/2)**2/np.sqrt(2)
            vel_set_x.value = a * (t-p/2) * 2/np.sqrt(2)
            vel_set_y.value = -a * (t-p/2) * 2/np.sqrt(2)
        elif 5*p/8 <= t < 3*p/4:
            pos_set_x.value = -0.5 * a * (t-3*p/4)**2/np.sqrt(2)
            pos_set_y.value = -l + 0.5 * a * (t-3*p/4)**2/np.sqrt(2)
            vel_set_x.value = a * (3*p/4-t) * 2/np.sqrt(2)
            vel_set_y.value = -a * (3*p/4-t) * 2/np.sqrt(2)
        elif 3*p/4 <= t < 7*p/8:
            pos_set_x.value = 0.5 * a * (t-3*p/4)**2/np.sqrt(2)
            pos_set_y.value = -l + 0.5 * a * (t-3*p/4)**2/np.sqrt(2)
            vel_set_x.value = a * (t-3*p/4) * 2/np.sqrt(2)
            vel_set_y.value = a * (t-3*p/4) * 2/np.sqrt(2)
        else:
            pos_set_x.value = l - 0.5 * a * (t-p)**2/np.sqrt(2)
            pos_set_y.value = -0.5 * a * (t-p)**2/np.sqrt(2)
            vel_set_x.value = a * (p-t) * 2/np.sqrt(2)
            vel_set_y.value = a * (p-t) * 2/np.sqrt(2)


if __name__ == '__main__':
    pos_set_x = Value('d', 0.0)
    pos_set_y = Value('d', 0.0)
    vel_set_x = Value('d', 0.0)
    vel_set_y = Value('d', 0.0)
    action_set = Array('d', [0.0, 0.0])
    real_pos_x = Value('d', 0.0)
    real_pos_y = Value('d', 0.0)
    vel_x = Value('d', 0.0)
    vel_y = Value('d', 0.0)
    plate_process = Process(target=PIDPlate, args=(action_set, real_pos_x, real_pos_y, pos_set_x, pos_set_y, vel_x, vel_y, vel_set_x, vel_set_y,))
    detect_process = Process(target=DetectBall, args=(action_set, real_pos_x, real_pos_y, pos_set_x, pos_set_y, vel_x, vel_y, vel_set_x, vel_set_y,))
    trajectory_process = Process(target=trajectory, args=(action_set, real_pos_x, real_pos_y, pos_set_x, pos_set_y, vel_x, vel_y, vel_set_x, vel_set_y,))
    plate_process.start()
    detect_process.start()
    trajectory_process.start()
    
    #Main process training 
    arr = [real_pos_x, real_pos_y, pos_set_x, pos_set_y, vel_x, vel_y, vel_set_x, vel_set_y]
    env = ballPlateEnv_v2.Ball_On_Plate_Robot_Env(position=arr)
    env.reset()
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_range = env.action_space.high  # scale action, [-action_range, action_range]
    agent = DDPG(action_dim, state_dim, action_range)
    t0 = time.time()
    all_episode_reward = []
    c_value = []
    for episode in range(TRAIN_EPISODES):
        state, _ = env.reset()
        episode_reward = 0
        for step in range(MAX_STEPS):
            # Add exploration noise
            action = agent.get_action(state)
            for i in range(2):
                action_set[i] = action[i]
            time.sleep(1)
            t = time.time()
            # print(action)
            state_, reward, done, _ = env.step(action)
            print(state-state_)
            # print(state_)
            
            agent.store_transition(state, action, reward, state_)
            c_value.append([action_set[0],action_set[1]])
            if agent.pointer > MEMORY_CAPACITY:
                print('action is:', action_set[0], action_set[1])
                agent.learn()
            state = state_
            episode_reward += reward
            if done:
                print('done')
                break
        if episode == 0:
            all_episode_reward.append(episode_reward)
        else:
            all_episode_reward.append(all_episode_reward[-1] * 0.9 + episode_reward * 0.1)
        
        print(
            'Training  | Episode: {}/{}  | Episode Reward: {:.4f}  | Running Time: {:.4f}'.format(
                episode + 1, TRAIN_EPISODES, episode_reward,
                time.time() - t0
            )
        )
    agent.save()
    env.close()
    plt.plot(all_episode_reward)
    if not os.path.exists('image'):
        os.makedirs('image')
    plt.savefig(os.path.join('image', '_'.join([ALG_NAME, ENV_ID])))
    plt.figure() #create new figure
    plt.plot([i[0] for i in c_value], label='c0')
    plt.plot([i[1] for i in c_value], label='c1')
    plt.xlabel('Timestep')
    plt.ylabel('Control parameters')
    plt.legend()
    plt.savefig(os.path.join('image', '_'.join(["1", ENV_ID])))
    print('training finished, please press "ctrl+c"')
    plate_process.join()
    detect_process.join()
    trajectory_process.join()